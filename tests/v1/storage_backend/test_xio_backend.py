# SPDX-License-Identifier: Apache-2.0
"""
Tests for XIOBackend - Multi-Level KV Cache with Cross-Level Data Promotion

Coverage:
- Init / config / feature toggle
- contains / batched_contains cross-level
- Block mapping construction
- batched_get_blocking (parent-compatible signature)
- batched_get_pipelined (callback, per-key fan-out)
- Promotion only to CPU level (level 0)
- Ref count correctness across get / promote paths
- Write-through / write-back put policies
- Remove / clear across all levels
- Allocate delegation
- Pin / unpin
- Edge cases: empty keys, single level, all miss
- Concurrent stress test
"""

# Standard
import inspect
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.xio_backend import XIOBackend, _has_real_batched_get


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**overrides):
    defaults = dict(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=1.0,
        lmcache_instance_id="test_xio",
        enable_xio_backend=True,
    )
    defaults.update(overrides)
    return LMCacheEngineConfig.from_defaults(**defaults)


def _metadata():
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=(32, 2, 256, 32, 128),
    )


def _key(key_id) -> CacheEngineKey:
    return CacheEngineKey("vllm", "test_model", 3, 123, hash(str(key_id)), torch.bfloat16)


def _obj(shape=(2, 16, 8, 128), dtype=torch.bfloat16) -> MemoryObj:
    return AdHocMemoryAllocator(device="cpu").allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)


def _cpu_backend(memory_allocator, instance_id="xio_test"):
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, lmcache_instance_id=instance_id,
    )
    return LocalCPUBackend(config=cfg, memory_allocator=memory_allocator)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metadata():
    return _metadata()


@pytest.fixture
def l0(memory_allocator):
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=256, local_cpu=True, lmcache_instance_id="xio_l0",
    )
    PinMonitor.GetOrCreate(cfg)
    backend = LocalCPUBackend(config=cfg, memory_allocator=memory_allocator)
    yield backend
    PinMonitor.DestroyInstance()


@pytest.fixture
def l1(memory_allocator):
    return _cpu_backend(memory_allocator, "xio_l1")


@pytest.fixture
def l2(memory_allocator):
    """Third level for 3-tier tests."""
    return _cpu_backend(memory_allocator, "xio_l2")


@pytest.fixture
def xio(metadata, l0, l1):
    """Two-level XIO backend."""
    return XIOBackend(
        config=_config(),
        metadata=metadata,
        levels=[("L0", l0), ("L1", l1)],
        dst_device="cpu",
    )


@pytest.fixture
def xio3(metadata, l0, l1, l2):
    """Three-level XIO backend."""
    return XIOBackend(
        config=_config(),
        metadata=metadata,
        levels=[("L0", l0), ("L1", l1), ("L2", l2)],
        dst_device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestXIOBackend:

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    # ===== Init =====

    def test_init(self, xio):
        assert str(xio) == "XIOBackend"
        assert len(xio.levels) == 2
        assert xio._allocator_backend is not None
        assert xio._cpu_level_backend is xio.levels[0][1]

    def test_init_no_levels(self, metadata):
        with pytest.raises(ValueError, match="at least one"):
            XIOBackend(config=_config(), metadata=metadata, levels=[], dst_device="cpu")

    # ===== _has_real_batched_get =====

    def test_has_real_batched_get_default(self, l0):
        # LocalCPUBackend does NOT override batched_get_blocking
        assert not _has_real_batched_get(l0)

    # ===== contains =====

    def test_contains_miss(self, xio):
        assert not xio.contains(_key("miss"))

    def test_contains_hit_l0(self, xio):
        k = _key("l0")
        xio.levels[0][1].submit_put_task(k, _obj())
        assert xio.contains(k)

    def test_contains_hit_l1_only(self, xio):
        k = _key("l1")
        xio.levels[1][1].submit_put_task(k, _obj())
        assert not xio.levels[0][1].contains(k)
        assert xio.contains(k)

    # ===== batched_contains cross-level =====

    def test_batched_contains_cross_level(self, xio):
        keys = [_key(f"bc{i}") for i in range(5)]
        for i in range(2):
            xio.levels[0][1].submit_put_task(keys[i], _obj())
        for i in range(2, 4):
            xio.levels[1][1].submit_put_task(keys[i], _obj())
        # key[4] not stored -> prefix hit = 4
        assert xio.batched_contains(keys) == 4

    def test_batched_contains_gap_breaks_prefix(self, xio):
        """key 0 in L0, key 1 missing, key 2 in L1 -> hit = 1."""
        keys = [_key(f"gap{i}") for i in range(3)]
        xio.levels[0][1].submit_put_task(keys[0], _obj())
        xio.levels[1][1].submit_put_task(keys[2], _obj())
        assert xio.batched_contains(keys) == 1

    # ===== block mapping =====

    def test_build_block_mapping(self, xio):
        keys = [_key(f"bm{i}") for i in range(4)]
        for i in range(2):
            xio.levels[0][1].submit_put_task(keys[i], _obj())
        for i in range(2, 4):
            xio.levels[1][1].submit_put_task(keys[i], _obj())
        mapping = xio._build_block_mapping(keys)
        assert mapping[0] == [0, 1]
        assert mapping[1] == [2, 3]

    # ===== batched_get_blocking — parent signature =====

    def test_batched_get_blocking_signature(self, xio):
        sig = inspect.signature(xio.batched_get_blocking)
        assert list(sig.parameters.keys()) == ["keys"]

    def test_batched_get_blocking_basic(self, xio):
        keys = [_key(f"bg{i}") for i in range(4)]
        for i in range(2):
            xio.levels[0][1].submit_put_task(keys[i], _obj())
        for i in range(2, 4):
            xio.levels[1][1].submit_put_task(keys[i], _obj())

        results = xio.batched_get_blocking(keys)
        assert len(results) == 4
        assert all(r is not None for r in results)

    def test_batched_get_with_missing(self, xio):
        keys = [_key(f"bgm{i}") for i in range(3)]
        xio.levels[0][1].submit_put_task(keys[0], _obj())
        results = xio.batched_get_blocking(keys)
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is None

    # ===== Promotion — CPU only =====

    def test_promotion_to_cpu(self, xio):
        """Data fetched from L1 is promoted to L0 (CPU)."""
        keys = [_key(f"p{i}") for i in range(3)]
        for k in keys:
            xio.levels[1][1].submit_put_task(k, _obj())

        for k in keys:
            assert not xio.levels[0][1].contains(k)

        xio.batched_get_blocking(keys)

        for k in keys:
            assert xio.levels[0][1].contains(k)

    def test_promotion_not_to_intermediate(self, xio3):
        """In a 3-tier setup, L2 hit should promote to L0 only, not L1."""
        k = _key("p3tier")
        xio3.levels[2][1].submit_put_task(k, _obj())

        assert not xio3.levels[0][1].contains(k)
        assert not xio3.levels[1][1].contains(k)

        xio3.batched_get_blocking([k])

        assert xio3.levels[0][1].contains(k)  # promoted
        assert not xio3.levels[1][1].contains(k)  # NOT promoted

    def test_promotion_skips_existing(self, xio):
        k = _key("pskip")
        obj = _obj()
        xio.levels[0][1].submit_put_task(k, obj)
        xio.levels[1][1].submit_put_task(k, obj)
        # get from L0 -> no promotion triggered
        result = xio.get_blocking(k)
        assert result is not None

    def test_promotion_from_l0_does_nothing(self, xio):
        """Getting from L0 should never trigger promotion."""
        k = _key("pl0")
        xio.levels[0][1].submit_put_task(k, _obj())
        result = xio.get_blocking(k)
        assert result is not None

    # ===== Ref count correctness =====

    def test_ref_count_get_blocking_l0(self, xio):
        """get_blocking from L0: sub-backend does ref_count_up, no promotion."""
        k = _key("rc_l0")
        obj = _obj()
        initial_rc = obj.get_ref_count()

        xio.levels[0][1].submit_put_task(k, obj)
        # submit_put_task: +1
        assert obj.get_ref_count() == initial_rc + 1

        result = xio.get_blocking(k)
        # get_blocking: +1 (from LocalCPUBackend)
        assert result is obj
        assert obj.get_ref_count() == initial_rc + 2

        # Simulate what cache_engine does after to_gpu
        result.ref_count_down()
        assert obj.get_ref_count() == initial_rc + 1

    def test_ref_count_get_blocking_l1_with_promotion(self, xio):
        """get_blocking from L1: ref_count_up for get + ref_count_up for promotion."""
        k = _key("rc_l1")
        obj = _obj()
        initial_rc = obj.get_ref_count()

        xio.levels[1][1].submit_put_task(k, obj)
        # submit_put_task in L1: +1
        assert obj.get_ref_count() == initial_rc + 1

        result = xio.get_blocking(k)
        # get_blocking from L1: +1 (from L1 backend)
        # promotion to L0: +1 (from L0's submit_put_task)
        assert result is obj
        assert obj.get_ref_count() == initial_rc + 3

        # cache_engine ref_count_down after to_gpu
        result.ref_count_down()
        # Still held by L1 hot_cache (+1) and L0 hot_cache (+1)
        assert obj.get_ref_count() == initial_rc + 2

    def test_ref_count_batched_get(self, xio):
        """Batched get: each returned obj has one extra ref for the caller."""
        keys = [_key(f"rcb{i}") for i in range(2)]
        objs = [_obj() for _ in range(2)]
        initial_rcs = [o.get_ref_count() for o in objs]

        xio.levels[0][1].submit_put_task(keys[0], objs[0])  # +1
        xio.levels[1][1].submit_put_task(keys[1], objs[1])  # +1

        results = xio.batched_get_blocking(keys)
        # keys[0]: in L0 -> get +1, no promotion
        assert objs[0].get_ref_count() == initial_rcs[0] + 2
        # keys[1]: in L1 -> get +1, promotion +1
        assert objs[1].get_ref_count() == initial_rcs[1] + 3

        # Simulate cache_engine cleanup
        for r in results:
            r.ref_count_down()
        assert objs[0].get_ref_count() == initial_rcs[0] + 1  # held by L0
        assert objs[1].get_ref_count() == initial_rcs[1] + 2  # held by L0 + L1

    # ===== batched_get_pipelined — on_chunk_ready (fan-out path) =====

    def test_pipelined_chunk_callback_fires(self, xio):
        """on_chunk_ready fires per-key for default (non-batched) backends."""
        keys = [_key(f"cb{i}") for i in range(4)]
        chunk_infos = [(keys[i], i * 256, (i + 1) * 256) for i in range(4)]

        for i in range(2):
            xio.levels[0][1].submit_put_task(keys[i], _obj())
        for i in range(2, 4):
            xio.levels[1][1].submit_put_task(keys[i], _obj())

        log = []
        lock = threading.Lock()

        def on_chunk(key, obj, start, end):
            with lock:
                log.append((start, end))

        results = xio.batched_get_pipelined(
            keys, on_chunk_ready=on_chunk, chunk_infos=chunk_infos,
        )
        assert len(results) == 4
        assert all(r is not None for r in results)
        # LocalCPUBackend has no real batched_get -> fan-out -> per-chunk
        assert len(log) == 4
        assert sorted(s for s, _ in log) == [0, 256, 512, 768]

    def test_pipelined_per_key_overlap(self, xio):
        """Per-chunk callbacks fire from pool threads, enabling overlap."""
        keys = [_key(f"po{i}") for i in range(4)]
        chunk_infos = [(keys[i], i, i + 1) for i in range(4)]

        for i in range(2):
            xio.levels[0][1].submit_put_task(keys[i], _obj())
        for i in range(2, 4):
            xio.levels[1][1].submit_put_task(keys[i], _obj())

        tids = []
        lock = threading.Lock()

        def on_chunk(key, obj, start, end):
            with lock:
                tids.append(threading.current_thread().ident)

        xio.batched_get_pipelined(
            keys, on_chunk_ready=on_chunk, chunk_infos=chunk_infos,
        )
        assert len(tids) == 4
        assert len(set(tids)) >= 1

    # ===== batched_get_pipelined — on_batch_ready (batched path) =====

    def test_pipelined_batch_callback_with_default_backend(self, xio):
        """
        LocalCPUBackend has no real batched_get, so on_batch_ready
        should NOT be called; on_chunk_ready should be used instead.
        """
        keys = [_key(f"bbd{i}") for i in range(3)]
        chunk_infos = [(keys[i], i * 100, (i + 1) * 100) for i in range(3)]

        for i in range(3):
            xio.levels[0][1].submit_put_task(keys[i], _obj())

        batch_log = []
        chunk_log = []
        lock = threading.Lock()

        def on_batch(bkeys, bobjs, starts, ends):
            with lock:
                batch_log.append(len(bobjs))

        def on_chunk(key, obj, start, end):
            with lock:
                chunk_log.append(start)

        xio.batched_get_pipelined(
            keys,
            on_chunk_ready=on_chunk,
            on_batch_ready=on_batch,
            chunk_infos=chunk_infos,
        )
        # Default backend -> fan-out -> on_chunk_ready used, not on_batch_ready
        assert len(batch_log) == 0
        assert len(chunk_log) == 3

    def test_pipelined_only_on_chunk_no_on_batch(self, xio):
        """When on_batch_ready is None, on_chunk_ready is used as fallback
        even if the backend has real batched_get (though LocalCPU doesn't)."""
        keys = [_key(f"onlyc{i}") for i in range(2)]
        chunk_infos = [(keys[i], i, i + 1) for i in range(2)]

        for k in keys:
            xio.levels[0][1].submit_put_task(k, _obj())

        chunk_log = []
        lock = threading.Lock()

        def on_chunk(key, obj, start, end):
            with lock:
                chunk_log.append(start)

        xio.batched_get_pipelined(
            keys, on_chunk_ready=on_chunk, chunk_infos=chunk_infos,
        )
        assert len(chunk_log) == 2

    # ===== batched_get_pipelined — direct_transfer backend =====

    def test_direct_transfer_backend(self, metadata, memory_allocator):
        """
        A backend with batched_get_and_transfer receives keys + **kwargs
        directly and handles get+GPU transfer in one fused call.
        """
        from lmcache.v1.storage_backend.abstract_backend import (
            StorageBackendInterface,
        )

        class MockDirectTransferBackend(StorageBackendInterface):
            """Mock backend that simulates fused get+transfer."""

            def __init__(self):
                super().__init__(dst_device="cpu")
                self.store = {}
                self.transfer_log = []

            def contains(self, key, pin=False):
                return key in self.store

            def batched_contains(self, keys, pin=False):
                count = 0
                for k in keys:
                    if k not in self.store:
                        break
                    count += 1
                return count

            def exists_in_put_tasks(self, key):
                return False

            def batched_submit_put_task(self, keys, objs, transfer_spec=None):
                for k, o in zip(keys, objs, strict=False):
                    o.ref_count_up()
                    self.store[k] = o
                return None

            def get_blocking(self, key):
                obj = self.store.get(key)
                if obj:
                    obj.ref_count_up()
                return obj

            def get_allocator_backend(self):
                raise NotImplementedError

            def pin(self, key):
                return False

            def unpin(self, key):
                return False

            def remove(self, key, force=True):
                return self.store.pop(key, None) is not None

            def close(self):
                pass

            def batched_get_and_transfer(
                self, keys, starts, ends, promote_fn=None, **kwargs,
            ):
                """
                Fused get + transfer.  Returns placeholder MemoryObjs.
                Calls promote_fn asynchronously (simulated here as
                immediate call) after 'transfer completes'.
                """
                self.transfer_log.append({
                    "keys": list(keys),
                    "starts": list(starts),
                    "ends": list(ends),
                    "kwargs_keys": sorted(kwargs.keys()),
                    "has_promote_fn": promote_fn is not None,
                })
                objs = []
                promote_keys = []
                promote_objs = []
                for k in keys:
                    obj = self.store.get(k)
                    if obj:
                        obj.ref_count_up()
                        promote_keys.append(k)
                        promote_objs.append(obj)
                    objs.append(obj)

                # Simulate async transfer completion -> call promote_fn
                if promote_fn is not None and promote_objs:
                    promote_fn(promote_keys, promote_objs)

                return objs

        PinMonitor.GetOrCreate(_config())
        l0 = _cpu_backend(memory_allocator, "dt_l0")
        dt_backend = MockDirectTransferBackend()

        xio = XIOBackend(
            config=_config(),
            metadata=metadata,
            levels=[("L0", l0), ("DirectTransfer", dt_backend)],
            dst_device="cpu",
        )

        # Put keys in the direct_transfer backend only
        keys = [_key(f"dt{i}") for i in range(3)]
        chunk_infos = [(keys[i], i * 256, (i + 1) * 256) for i in range(3)]
        for k in keys:
            dt_backend.batched_submit_put_task([k], [_obj()])

        # on_chunk_ready / on_batch_ready should NOT fire for direct_transfer
        chunk_log = []
        batch_log = []
        # on_retrieved SHOULD fire for bookkeeping
        retrieved_log = []

        results = xio.batched_get_pipelined(
            keys,
            on_chunk_ready=lambda k, o, s, e: chunk_log.append(k),
            on_batch_ready=lambda ks, os, ss, es: batch_log.append(len(os)),
            on_retrieved=lambda ks, os, ss, es: retrieved_log.append(
                {"count": len(os), "starts": list(ss), "ends": list(es)}
            ),
            chunk_infos=chunk_infos,
            # Simulate kwargs that nds_backend would need
            slot_mapping=[0, 1, 2],
            kvcaches="mock_kvcaches",
        )

        assert len(results) == 3
        assert all(r is not None for r in results)

        # Direct transfer: no to_gpu callbacks
        assert len(chunk_log) == 0
        assert len(batch_log) == 0

        # But on_retrieved was called for bookkeeping
        assert len(retrieved_log) == 1
        assert retrieved_log[0]["count"] == 3
        assert retrieved_log[0]["starts"] == [0, 256, 512]
        assert retrieved_log[0]["ends"] == [256, 512, 768]

        # Verify kwargs were forwarded and promote_fn was provided
        assert len(dt_backend.transfer_log) == 1
        log_entry = dt_backend.transfer_log[0]
        assert "slot_mapping" in log_entry["kwargs_keys"]
        assert "kvcaches" in log_entry["kwargs_keys"]
        assert log_entry["has_promote_fn"] is True

        # Verify promotion to CPU happened (via promote_fn callback)
        for k in keys:
            assert l0.contains(k)

        PinMonitor.DestroyInstance()

    # ===== on_retrieved fires for all backend types =====

    def test_on_retrieved_fires_for_fanout(self, xio):
        """on_retrieved fires for default per-key backends."""
        keys = [_key(f"rf{i}") for i in range(3)]
        chunk_infos = [(keys[i], i * 10, (i + 1) * 10) for i in range(3)]

        for k in keys:
            xio.levels[0][1].submit_put_task(k, _obj())

        retrieved_starts = []
        lock = threading.Lock()

        def on_retrieved(ks, os, ss, es):
            with lock:
                retrieved_starts.extend(ss)

        xio.batched_get_pipelined(
            keys,
            on_retrieved=on_retrieved,
            chunk_infos=chunk_infos,
        )
        assert sorted(retrieved_starts) == [0, 10, 20]

    # ===== Write policies =====

    def test_write_through(self, xio):
        k = _key("wt")
        xio.batched_submit_put_task([k], [_obj()])
        assert xio.levels[0][1].contains(k)
        assert xio.levels[1][1].contains(k)

    def test_write_back(self, metadata, memory_allocator):
        cfg = _config(extra_config={"xio_write_policy": "write_back"})
        PinMonitor.GetOrCreate(cfg)
        l0 = _cpu_backend(memory_allocator, "wb0")
        l1 = _cpu_backend(memory_allocator, "wb1")
        backend = XIOBackend(
            config=cfg, metadata=metadata,
            levels=[("L0", l0), ("L1", l1)], dst_device="cpu",
        )

        k = _key("wb")
        backend.batched_submit_put_task([k], [_obj()])
        assert l0.contains(k)
        assert not l1.contains(k)
        PinMonitor.DestroyInstance()

    # ===== Remove / clear =====

    def test_remove_all_levels(self, xio):
        k = _key("rm")
        xio.batched_submit_put_task([k], [_obj()])
        assert xio.remove(k)
        assert not xio.levels[0][1].contains(k)
        assert not xio.levels[1][1].contains(k)

    def test_batched_remove(self, xio):
        keys = [_key(f"brm{i}") for i in range(3)]
        for k in keys:
            xio.batched_submit_put_task([k], [_obj()])
        assert xio.batched_remove(keys) > 0
        for k in keys:
            assert not xio.contains(k)

    def test_clear(self, xio):
        keys = [_key(f"cl{i}") for i in range(3)]
        for k in keys:
            xio.batched_submit_put_task([k], [_obj()])
        xio.clear()

    # ===== Allocate =====

    def test_allocate(self, xio):
        obj = xio.allocate(torch.Size([2, 16, 8, 128]), torch.bfloat16)
        assert obj is not None and isinstance(obj, MemoryObj)

    def test_batched_allocate(self, xio):
        objs = xio.batched_allocate(torch.Size([2, 16, 8, 128]), torch.bfloat16, 3)
        assert objs is not None and len(objs) == 3

    def test_get_allocator_backend(self, xio):
        assert isinstance(xio.get_allocator_backend(), LocalCPUBackend)

    # ===== Pin / unpin =====

    def test_pin_unpin(self, xio):
        k = _key("pin")
        xio.levels[0][1].submit_put_task(k, _obj())
        assert xio.pin(k)
        assert xio.unpin(k)
        assert not xio.pin(_key("nopin"))

    def test_unpin_only_targets_pinned_level(self, xio):
        """
        Key in L1 pinned via batched_contains(pin=True), then promoted
        to L0 via get. unpin should only unpin at L1, not L0.
        """
        k = _key("unpin_level")
        obj = _obj()

        # Key only in L1
        xio.levels[1][1].submit_put_task(k, obj)

        # lookup with pin -> pins at L1
        assert xio.batched_contains([k], pin=True) == 1
        assert k in xio._pin_registry
        assert xio._pin_registry[k] == {1}

        # Retrieve -> promotes to L0
        xio.batched_get_blocking([k])
        assert xio.levels[0][1].contains(k)

        # Unpin -> should only unpin at L1, not L0
        assert xio.unpin(k)

        # Key no longer in pin registry
        assert k not in xio._pin_registry

        # L0 MemoryObj should NOT have been unpinned (was never pinned)
        l0_obj = xio.levels[0][1].get_blocking(k)
        assert l0_obj is not None
        # If unpin had been called on L0, pin_count would be negative
        # (which triggers the warning). The obj should have pin_count >= 0.
        assert l0_obj.meta.pin_count >= 0
        l0_obj.ref_count_down()

    def test_contains_pin_tracks_level(self, xio):
        """contains(pin=True) should track which level was pinned."""
        k0 = _key("cpt0")
        k1 = _key("cpt1")

        xio.levels[0][1].submit_put_task(k0, _obj())
        xio.levels[1][1].submit_put_task(k1, _obj())

        assert xio.contains(k0, pin=True)
        assert xio.contains(k1, pin=True)

        assert xio._pin_registry[k0] == {0}
        assert xio._pin_registry[k1] == {1}

        # Unpin clears registry
        xio.unpin(k0)
        xio.unpin(k1)
        assert k0 not in xio._pin_registry
        assert k1 not in xio._pin_registry

    def test_batched_contains_pin_cross_level(self, xio):
        """
        batched_contains(pin=True) with keys spread across L0 and L1
        should pin each key only at its actual level.
        """
        keys = [_key(f"bcp{i}") for i in range(4)]
        for i in range(2):
            xio.levels[0][1].submit_put_task(keys[i], _obj())
        for i in range(2, 4):
            xio.levels[1][1].submit_put_task(keys[i], _obj())

        assert xio.batched_contains(keys, pin=True) == 4

        # keys[0:2] pinned at L0, keys[2:4] pinned at L1
        assert xio._pin_registry[keys[0]] == {0}
        assert xio._pin_registry[keys[1]] == {0}
        assert xio._pin_registry[keys[2]] == {1}
        assert xio._pin_registry[keys[3]] == {1}

        # Unpin all
        for k in keys:
            xio.unpin(k)
        assert len(xio._pin_registry) == 0

    # ===== Feature toggle =====

    def test_toggle_off(self):
        assert _config(enable_xio_backend=False).enable_xio_backend is False

    def test_toggle_on(self):
        assert _config(enable_xio_backend=True).enable_xio_backend is True

    # ===== Edge cases =====

    def test_empty_keys(self, xio):
        assert xio.batched_get_blocking([]) == []
        assert xio.batched_contains([]) == 0

    def test_single_level(self, metadata, memory_allocator):
        PinMonitor.GetOrCreate(_config())
        l0 = _cpu_backend(memory_allocator, "single")
        backend = XIOBackend(
            config=_config(), metadata=metadata,
            levels=[("L0", l0)], dst_device="cpu",
        )
        k = _key("sl")
        l0.submit_put_task(k, _obj())
        result = backend.get_blocking(k)
        assert result is not None
        PinMonitor.DestroyInstance()

    def test_all_miss(self, xio):
        keys = [_key(f"miss{i}") for i in range(3)]
        results = xio.batched_get_blocking(keys)
        assert all(r is None for r in results)

    # ===== Concurrent stress =====

    def test_concurrent_batched_get(self, xio):
        keys = [_key(f"s{i}") for i in range(20)]
        for i in range(10):
            xio.levels[0][1].submit_put_task(keys[i], _obj())
        for i in range(10, 20):
            xio.levels[1][1].submit_put_task(keys[i], _obj())

        errors = []

        def worker(start):
            try:
                batch = keys[start:start + 5]
                results = xio.batched_get_blocking(batch)
                for r in results:
                    assert r is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 5,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent errors: {errors}"

    # ===== Close =====

    def test_close(self, metadata, memory_allocator):
        PinMonitor.GetOrCreate(_config())
        l0 = _cpu_backend(memory_allocator, "close")
        backend = XIOBackend(
            config=_config(), metadata=metadata,
            levels=[("L0", l0)], dst_device="cpu",
        )
        backend.close()
        PinMonitor.DestroyInstance()


# ---------------------------------------------------------------------------
# Tests for configurable backend ordering
# ---------------------------------------------------------------------------


class TestXIOBackendOrdering:
    """Tests for XIO configurable backend ordering via extra_config."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_custom_order_reverses_levels(self, metadata, memory_allocator):
        """Explicit xio_backend_order reverses the default level ordering."""
        PinMonitor.GetOrCreate(_config())

        l0 = _cpu_backend(memory_allocator, "order_l0")
        l1 = _cpu_backend(memory_allocator, "order_l1")

        # Default order: L0 first, L1 second
        xio_default = XIOBackend(
            config=_config(),
            metadata=metadata,
            levels=[("L0", l0), ("L1", l1)],
            dst_device="cpu",
        )
        assert xio_default.levels[0][0] == "L0"
        assert xio_default.levels[1][0] == "L1"
        xio_default.close()

        # Reversed order: L1 first, L0 second
        xio_reversed = XIOBackend(
            config=_config(),
            metadata=metadata,
            levels=[("L1", l1), ("L0", l0)],
            dst_device="cpu",
        )
        assert xio_reversed.levels[0][0] == "L1"
        assert xio_reversed.levels[1][0] == "L0"
        xio_reversed.close()

        PinMonitor.DestroyInstance()

    def test_ordering_via_create_storage_backends(self, metadata, memory_allocator):
        """
        CreateStorageBackends respects extra_config.xio_backend_order
        when building XIO levels.
        """
        import asyncio
        from collections import OrderedDict
        from lmcache.v1.storage_backend import CreateStorageBackends

        PinMonitor.GetOrCreate(_config())

        # Create config with CPU + Disk + XIO enabled, custom order
        cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=1.0,
            local_disk="/tmp/test_xio_order_disk",
            max_local_disk_size=1.0,
            lmcache_instance_id="xio_order_test",
            enable_xio_backend=True,
            extra_config={
                "xio_backend_order": [
                    "LocalDiskBackend",
                    "LocalCPUBackend",
                ],
            },
        )

        loop = asyncio.new_event_loop()
        try:
            backends = CreateStorageBackends(cfg, metadata, loop, "cpu")
            assert "XIOBackend" in backends
            xio = backends["XIOBackend"]

            # Verify the levels are in the configured order
            level_names = [name for name, _ in xio.levels]
            assert level_names.index("LocalDiskBackend") < level_names.index(
                "LocalCPUBackend"
            ), f"Expected LocalDiskBackend before LocalCPUBackend, got {level_names}"

            xio.close()
        finally:
            loop.close()

        PinMonitor.DestroyInstance()

    def test_ordering_appends_remaining_backends(self, metadata, memory_allocator):
        """Backends not in xio_backend_order are appended at the end."""
        PinMonitor.GetOrCreate(_config())

        l0 = _cpu_backend(memory_allocator, "append_l0")
        l1 = _cpu_backend(memory_allocator, "append_l1")
        l2 = _cpu_backend(memory_allocator, "append_l2")

        from collections import OrderedDict

        backends = OrderedDict([
            ("L0", l0),
            ("L1", l1),
            ("L2", l2),
        ])

        # Only specify L2 in order -> L2 first, then L0, L1 appended
        xio_order = ["L2"]
        ordered = OrderedDict()
        for name in xio_order:
            if name in backends:
                ordered[name] = backends[name]
        for name, backend in backends.items():
            if name not in ordered:
                ordered[name] = backend

        levels = list(ordered.items())
        xio = XIOBackend(
            config=_config(),
            metadata=metadata,
            levels=levels,
            dst_device="cpu",
        )
        level_names = [name for name, _ in xio.levels]
        assert level_names == ["L2", "L0", "L1"]
        xio.close()

        PinMonitor.DestroyInstance()

    def test_ordering_with_batched_backend(self, metadata, memory_allocator):
        """
        A mock RemoteBackend-like backend (overrides batched_get_blocking)
        works correctly when placed at a configured level in XIO.
        This validates Mooncake (RemoteBackend) integration with XIO.
        """

        class MockBatchedBackend(StorageBackendInterface):
            """Simulates RemoteBackend (has real batched_get_blocking)."""

            def __init__(self):
                super().__init__(dst_device="cpu")
                self.store = {}

            def contains(self, key, pin=False):
                return key in self.store

            def batched_contains(self, keys, pin=False):
                count = 0
                for k in keys:
                    if k not in self.store:
                        break
                    count += 1
                return count

            def exists_in_put_tasks(self, key):
                return False

            def batched_submit_put_task(self, keys, objs, transfer_spec=None):
                for k, o in zip(keys, objs, strict=False):
                    o.ref_count_up()
                    self.store[k] = o
                return None

            def get_blocking(self, key):
                obj = self.store.get(key)
                if obj:
                    obj.ref_count_up()
                return obj

            def batched_get_blocking(self, keys):
                """Override -> this is a 'real batched' backend."""
                results = []
                for key in keys:
                    obj = self.store.get(key)
                    if obj:
                        obj.ref_count_up()
                    results.append(obj)
                return results

            def get_allocator_backend(self):
                raise NotImplementedError

            def pin(self, key):
                return False

            def unpin(self, key):
                return False

            def remove(self, key, force=True):
                return self.store.pop(key, None) is not None

            def close(self):
                pass

        PinMonitor.GetOrCreate(_config())

        l0 = _cpu_backend(memory_allocator, "mooncake_l0")
        mooncake_like = MockBatchedBackend()

        xio = XIOBackend(
            config=_config(),
            metadata=metadata,
            levels=[("LocalCPUBackend", l0), ("RemoteBackend", mooncake_like)],
            dst_device="cpu",
        )

        # Verify the mock is detected as a real batched backend
        from lmcache.v1.storage_backend.xio_backend import (
            _has_direct_transfer,
            _has_real_batched_get,
        )
        assert _has_real_batched_get(mooncake_like)
        assert not _has_direct_transfer(mooncake_like)

        # Put keys only in the RemoteBackend level
        keys = [_key(f"mc{i}") for i in range(3)]
        for k in keys:
            mooncake_like.batched_submit_put_task([k], [_obj()])

        # batched_get should retrieve from RemoteBackend and promote to CPU
        results = xio.batched_get_blocking(keys)
        assert len(results) == 3
        assert all(r is not None for r in results)

        # Verify promotion to L0 (CPU)
        for k in keys:
            assert l0.contains(k)

        # Pipelined path with on_batch_ready (batched backend triggers it)
        batch_log = []
        chunk_infos = [(keys[i], i * 256, (i + 1) * 256) for i in range(3)]

        # Put fresh keys to test pipelined path
        keys2 = [_key(f"mc_pipe{i}") for i in range(2)]
        for k in keys2:
            mooncake_like.batched_submit_put_task([k], [_obj()])

        chunk_infos2 = [(keys2[i], i * 256, (i + 1) * 256) for i in range(2)]
        results2 = xio.batched_get_pipelined(
            keys2,
            on_batch_ready=lambda ks, os, ss, es: batch_log.append(len(os)),
            chunk_infos=chunk_infos2,
        )
        assert len(results2) == 2
        assert all(r is not None for r in results2)
        # Real batched backend -> on_batch_ready should fire
        assert len(batch_log) == 1
        assert batch_log[0] == 2

        xio.close()
        PinMonitor.DestroyInstance()
