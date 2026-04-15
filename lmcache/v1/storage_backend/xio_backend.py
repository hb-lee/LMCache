# SPDX-License-Identifier: Apache-2.0
"""
XIO Backend - Multi-Level KV Cache with Cross-Level Data Promotion

Wraps multiple storage backends into a single hierarchical cache.
When a key is found at a lower (slower) level but not at the CPU level
(level 0), the data is automatically promoted to CPU for faster future
access.

Design principles:
- ``batched_get_blocking`` keeps the parent-class signature (keys only).
- ``batched_get_pipelined`` is a *new* method that accepts an
  ``on_chunk_ready`` callback for pipelined get + to_gpu.
- When a sub-backend has a *real* ``batched_get_blocking`` override
  (e.g. GDS, Remote), the batch is forwarded as-is.
- When a sub-backend uses the *default* per-key loop, the keys are
  fanned out to the thread pool so that different keys' get + to_gpu
  can overlap in a pipeline fashion.
- Promotion only targets the CPU level (level 0), not intermediate
  levels like disk or GDS.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)

logger = init_logger(__name__)

# Callback fired per-chunk for to_gpu (fan-out pipeline):
#   (key, memory_obj, start, end) -> None
OnChunkReadyCallback = Callable[
    [CacheEngineKey, MemoryObj, int, int], None
]

# Callback fired per-batch for to_gpu (batched backends):
#   (keys, memory_objs, starts, ends) -> None
OnBatchReadyCallback = Callable[
    [List[CacheEngineKey], List[MemoryObj], List[int], List[int]], None
]

# Callback fired for every retrieved chunk/batch to update common
# bookkeeping (ret_mask, tot_kv_size, reordered_chunks).
# Signature: (keys, memory_objs, starts, ends) -> None
OnRetrievedCallback = Callable[
    [List[CacheEngineKey], List[MemoryObj], List[int], List[int]], None
]


def _has_real_batched_get(backend: StorageBackendInterface) -> bool:
    """
    Return True if ``backend`` overrides ``batched_get_blocking``
    with its own implementation (i.e. not the default per-key loop
    from ``StorageBackendInterface``).
    """
    return (
        type(backend).batched_get_blocking
        is not StorageBackendInterface.batched_get_blocking
    )


def _has_direct_transfer(backend: StorageBackendInterface) -> bool:
    """
    Return True if ``backend`` exposes ``batched_get_and_transfer``,
    a fused method that performs get + GPU transfer in one call and
    requires ``**kwargs`` (kv pointers, slot_mapping, etc.).
    """
    return hasattr(backend, "batched_get_and_transfer") and callable(
        getattr(backend, "batched_get_and_transfer")
    )


class XIOBackend(AllocatorBackendInterface):
    """
    A multi-level cache backend that manages an ordered list of sub-backends
    (levels) from fastest (level 0) to slowest (level N).

    Key behaviours:
    - **contains/lookup**: checks levels top-down, returns on first hit.
    - **get**: retrieves from the hit level(s) and promotes to CPU (level 0)
      if the data was found at a lower level.
    - **put**: writes to all levels (write-through) by default.
    - **remove**: removes from all levels.
    - **allocate**: delegates to the allocator backend (level 0).
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        levels: List[Tuple[str, StorageBackendInterface]],
        loop: Optional[asyncio.AbstractEventLoop] = None,
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device=dst_device)
        self.config = config
        self.metadata = metadata
        self.loop = loop

        if not levels:
            raise ValueError("XIOBackend requires at least one sub-backend level")

        # Ordered list of (name, backend) from fastest to slowest
        self.levels: List[Tuple[str, StorageBackendInterface]] = levels

        # Find the allocator backend (first AllocatorBackendInterface)
        self._allocator_backend: Optional[AllocatorBackendInterface] = None
        for name, backend in self.levels:
            if isinstance(backend, AllocatorBackendInterface):
                self._allocator_backend = backend
                break

        if self._allocator_backend is None:
            raise ValueError(
                "XIOBackend requires at least one AllocatorBackendInterface "
                "sub-backend for memory allocation"
            )

        # CPU backend (level 0) used as promotion target
        self._cpu_level_name, self._cpu_level_backend = self.levels[0]

        # Track which level(s) each key was pinned at during
        # batched_contains(pin=True), so that unpin only targets
        # the correct levels.
        # key -> set of level indices where pin was applied
        self._pin_registry: Dict[CacheEngineKey, set] = {}
        self._pin_registry_lock = threading.Lock()

        # Write policy: "write_through" (default) or "write_back"
        self._write_policy = "write_through"
        if config.extra_config:
            self._write_policy = config.extra_config.get(
                "xio_write_policy", "write_through"
            )

        # Thread pool for concurrent gets (cross-level + per-key fan-out)
        default_workers = max(4, len(self.levels) * 2)
        if config.extra_config:
            default_workers = config.extra_config.get(
                "xio_get_workers", default_workers
            )
        self._get_pool = ThreadPoolExecutor(
            max_workers=default_workers,
            thread_name_prefix="xio-get",
        )

        logger.info(
            "XIOBackend initialized with %d levels: %s, write_policy=%s, "
            "get_workers=%d",
            len(self.levels),
            [name for name, _ in self.levels],
            self._write_policy,
            default_workers,
        )

    def __str__(self):
        return "XIOBackend"

    # ------------------------------------------------------------------
    # Block mapping
    # ------------------------------------------------------------------

    def _build_block_mapping(
        self,
        keys: List[CacheEngineKey],
    ) -> Dict[int, List[int]]:
        """
        For each key (by prefix matching), find which level it lives in.

        Returns:
            dict mapping level_index -> list of *original indices* in
            ``keys`` that are found at that level.  Only prefix-contiguous
            hits are returned (the first miss breaks the chain).
        """
        block_mapping: Dict[int, List[int]] = {}
        remaining = list(range(len(keys)))

        while remaining:
            found_any = False
            for level_idx, (_, backend) in enumerate(self.levels):
                remaining_keys = [keys[i] for i in remaining]
                hit_count = backend.batched_contains(remaining_keys)
                if hit_count == 0:
                    continue

                indices = remaining[:hit_count]
                block_mapping.setdefault(level_idx, []).extend(indices)
                remaining = remaining[hit_count:]
                found_any = True
                break  # restart from top level for remaining keys

            if not found_any:
                break

        return block_mapping

    # ------------------------------------------------------------------
    # contains
    # ------------------------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        for level_idx, (_, backend) in enumerate(self.levels):
            if backend.contains(key, pin):
                if pin:
                    with self._pin_registry_lock:
                        self._pin_registry.setdefault(key, set()).add(level_idx)
                return True
        return False

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Prefix-match across all levels combined."""
        hit_chunks = 0
        remaining_keys = keys
        while remaining_keys:
            found_any = False
            for level_idx, (_, backend) in enumerate(self.levels):
                count = backend.batched_contains(remaining_keys, pin)
                if count > 0:
                    if pin:
                        with self._pin_registry_lock:
                            for k in remaining_keys[:count]:
                                self._pin_registry.setdefault(k, set()).add(
                                    level_idx
                                )
                    hit_chunks += count
                    remaining_keys = remaining_keys[count:]
                    found_any = True
                    break
            if not found_any:
                break
        return hit_chunks

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Async prefix-match across all levels."""
        # Delegate to the sync version which handles pin tracking
        return self.batched_contains(keys, pin)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        for _, backend in self.levels:
            if backend.exists_in_put_tasks(key):
                return True
        return False

    # ------------------------------------------------------------------
    # get  —  parent-compatible signature
    # ------------------------------------------------------------------

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Get from the first level that contains the key.
        If found at level N > 0, promote to CPU (level 0).
        """
        for level_idx, (_, backend) in enumerate(self.levels):
            memory_obj = backend.get_blocking(key)
            if memory_obj is not None:
                if level_idx > 0:
                    self._promote_to_cpu([key], [memory_obj])
                return memory_obj
        return None

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        Parent-compatible batched get (signature matches
        ``StorageBackendInterface.batched_get_blocking``).

        Dispatches batch-gets to each level concurrently and merges
        results.
        """
        return self._do_batched_get(
            keys,
            on_chunk_ready=None,
            on_batch_ready=None,
            on_retrieved=None,
            chunk_infos=None,
        )

    # ------------------------------------------------------------------
    # get  —  pipelined variant (new method, not an override)
    # ------------------------------------------------------------------

    def batched_get_pipelined(
        self,
        keys: List[CacheEngineKey],
        on_chunk_ready: Optional[OnChunkReadyCallback] = None,
        on_batch_ready: Optional[OnBatchReadyCallback] = None,
        on_retrieved: Optional[OnRetrievedCallback] = None,
        chunk_infos: Optional[List[Tuple[CacheEngineKey, int, int]]] = None,
        **kwargs,
    ) -> List[Optional[MemoryObj]]:
        """
        Pipelined batched get with adaptive callback strategy.

        Callbacks:
        - ``on_retrieved``: **always** called for every successfully
          retrieved chunk, regardless of backend type.  Use this to
          update common bookkeeping (ret_mask, tot_kv_size, etc.).
        - ``on_chunk_ready``: fired per-key for to_gpu (fan-out path).
        - ``on_batch_ready``: fired per-batch for to_gpu (batched path).
        - For direct_transfer backends, neither on_chunk_ready nor
          on_batch_ready is called (get+transfer is fused), but
          on_retrieved is still called.

        :param kwargs: Forwarded to backends with fused get+transfer.
        """
        return self._do_batched_get(
            keys, on_chunk_ready, on_batch_ready, on_retrieved,
            chunk_infos, **kwargs,
        )

    # ------------------------------------------------------------------
    # get  —  shared implementation
    # ------------------------------------------------------------------

    def _do_batched_get(
        self,
        keys: List[CacheEngineKey],
        on_chunk_ready: Optional[OnChunkReadyCallback],
        on_batch_ready: Optional[OnBatchReadyCallback],
        on_retrieved: Optional[OnRetrievedCallback],
        chunk_infos: Optional[List[Tuple[CacheEngineKey, int, int]]],
        **kwargs,
    ) -> List[Optional[MemoryObj]]:
        block_mapping = self._build_block_mapping(keys)
        if not block_mapping:
            return [None] * len(keys)

        result_map: Dict[int, MemoryObj] = {}
        result_lock = threading.Lock()

        def _get_start_end(idx: int) -> Tuple[int, int]:
            if chunk_infos is not None:
                _, start, end = chunk_infos[idx]
                return start, end
            return idx, idx + 1

        def _record(idx: int, obj: MemoryObj):
            with result_lock:
                result_map[idx] = obj

        def _notify_retrieved(
            ok_keys: List[CacheEngineKey],
            ok_objs: List[MemoryObj],
            ok_starts: List[int],
            ok_ends: List[int],
        ):
            """Common bookkeeping callback for all paths."""
            if on_retrieved is not None and ok_objs:
                on_retrieved(ok_keys, ok_objs, ok_starts, ok_ends)

        # -- per-level dispatchers --

        def _fetch_direct_transfer(
            level_idx: int,
            indices: List[int],
            backend: StorageBackendInterface,
        ):
            """
            Backend has ``batched_get_and_transfer`` -> fused get+GPU
            transfer in one call.  No on_chunk_ready / on_batch_ready.

            The backend receives a ``promote_fn(keys, memory_objs)``
            callback.  Since the transfer is asynchronous, the backend
            allocates placeholder MemoryObjs (with correct metadata but
            no content) and returns them immediately.  When the async
            transfer completes, the backend calls ``promote_fn`` with
            the filled MemoryObjs to write them into the CPU cache.
            """
            level_keys = [keys[i] for i in indices]
            level_starts = [_get_start_end(i)[0] for i in indices]
            level_ends = [_get_start_end(i)[1] for i in indices]

            # Build promote callback for the backend to call after
            # async transfer completes and MemoryObj content is ready.
            def promote_fn(
                p_keys: List[CacheEngineKey],
                p_objs: List[MemoryObj],
            ) -> None:
                self._promote_to_cpu(p_keys, p_objs)

            memory_objs = backend.batched_get_and_transfer(
                level_keys, level_starts, level_ends,
                promote_fn=promote_fn if level_idx > 0 else None,
                **kwargs,
            )

            ok_keys: List[CacheEngineKey] = []
            ok_objs: List[MemoryObj] = []
            ok_starts: List[int] = []
            ok_ends: List[int] = []

            if memory_objs:
                for idx, mem_obj in zip(indices, memory_objs, strict=False):
                    if mem_obj is None:
                        continue
                    _record(idx, mem_obj)
                    ok_keys.append(keys[idx])
                    ok_objs.append(mem_obj)
                    ok_starts.append(_get_start_end(idx)[0])
                    ok_ends.append(_get_start_end(idx)[1])

            _notify_retrieved(ok_keys, ok_objs, ok_starts, ok_ends)

        def _fetch_batched(
            level_idx: int,
            indices: List[int],
            backend: StorageBackendInterface,
        ):
            """
            Backend has real batched_get_blocking -> forward the whole
            batch.  Fire on_batch_ready for batched_to_gpu, or fall
            back to per-chunk on_chunk_ready.
            """
            level_keys = [keys[i] for i in indices]
            memory_objs = backend.batched_get_blocking(level_keys)

            ok_keys: List[CacheEngineKey] = []
            ok_objs: List[MemoryObj] = []
            ok_starts: List[int] = []
            ok_ends: List[int] = []

            for idx, mem_obj in zip(indices, memory_objs, strict=False):
                if mem_obj is None:
                    continue
                _record(idx, mem_obj)
                start, end = _get_start_end(idx)
                ok_keys.append(keys[idx])
                ok_objs.append(mem_obj)
                ok_starts.append(start)
                ok_ends.append(end)

            _notify_retrieved(ok_keys, ok_objs, ok_starts, ok_ends)

            if ok_objs:
                if on_batch_ready is not None:
                    on_batch_ready(ok_keys, ok_objs, ok_starts, ok_ends)
                elif on_chunk_ready is not None:
                    for k, o, s, e in zip(
                        ok_keys, ok_objs, ok_starts, ok_ends, strict=False,
                    ):
                        on_chunk_ready(k, o, s, e)

            if level_idx > 0 and ok_objs:
                self._promote_to_cpu(ok_keys, ok_objs)

        def _fetch_single_key(
            level_idx: int,
            idx: int,
            backend: StorageBackendInterface,
        ) -> Optional[Tuple[CacheEngineKey, MemoryObj]]:
            """Fetch one key, record, notify, fire per-chunk callback."""
            key = keys[idx]
            mem_obj = backend.get_blocking(key)
            if mem_obj is None:
                return None
            _record(idx, mem_obj)
            start, end = _get_start_end(idx)
            _notify_retrieved([key], [mem_obj], [start], [end])
            if on_chunk_ready is not None:
                on_chunk_ready(key, mem_obj, start, end)
            return (key, mem_obj)

        def _fetch_fanout(
            level_idx: int,
            indices: List[int],
            backend: StorageBackendInterface,
        ):
            """
            Default per-key backend -> fan out each key to the thread
            pool so get + per-chunk callback (to_gpu) of different keys
            overlap in a pipeline.
            """
            futures = [
                self._get_pool.submit(
                    _fetch_single_key, level_idx, idx, backend,
                )
                for idx in indices
            ]

            promoted_keys: List[CacheEngineKey] = []
            promoted_objs: List[MemoryObj] = []

            for fut in as_completed(futures):
                result = fut.result()
                if result is not None and level_idx > 0:
                    promoted_keys.append(result[0])
                    promoted_objs.append(result[1])

            if promoted_objs:
                self._promote_to_cpu(promoted_keys, promoted_objs)

        # -- dispatch each level concurrently --
        level_futures: List[Future] = []
        for level_idx, indices in block_mapping.items():
            _, backend = self.levels[level_idx]

            if _has_direct_transfer(backend):
                # Fused get+transfer, kwargs forwarded
                fut = self._get_pool.submit(
                    _fetch_direct_transfer, level_idx, indices, backend,
                )
            elif _has_real_batched_get(backend):
                # Batched get -> batched callback
                fut = self._get_pool.submit(
                    _fetch_batched, level_idx, indices, backend,
                )
            else:
                # Default per-key -> fan-out pipeline
                fut = self._get_pool.submit(
                    _fetch_fanout, level_idx, indices, backend,
                )
            level_futures.append(fut)

        for fut in level_futures:
            fut.result()

        return [result_map.get(i) for i in range(len(keys))]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Async batched get.  Dispatches to each level concurrently."""
        block_mapping = self._build_block_mapping(keys)
        if not block_mapping:
            return []

        result_map: Dict[int, MemoryObj] = {}

        async def _fetch_level(level_idx: int, indices: List[int]):
            _, backend = self.levels[level_idx]
            level_keys = [keys[i] for i in indices]

            try:
                objs = await backend.batched_get_non_blocking(
                    lookup_id, level_keys, transfer_spec,
                )
            except NotImplementedError:
                objs = await asyncio.get_event_loop().run_in_executor(
                    self._get_pool,
                    backend.batched_get_blocking,
                    level_keys,
                )

            promoted_keys: List[CacheEngineKey] = []
            promoted_objs: List[MemoryObj] = []

            for idx, obj in zip(indices, objs, strict=False):
                if obj is None:
                    continue
                result_map[idx] = obj
                if level_idx > 0:
                    promoted_keys.append(keys[idx])
                    promoted_objs.append(obj)

            if promoted_objs:
                self._promote_to_cpu(promoted_keys, promoted_objs)

        tasks = [
            asyncio.create_task(_fetch_level(level_idx, indices))
            for level_idx, indices in block_mapping.items()
        ]
        await asyncio.gather(*tasks)

        mem_objs: list[MemoryObj] = []
        for i in range(len(keys)):
            if i in result_map:
                mem_objs.append(result_map[i])
            else:
                break
        return mem_objs

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        for _, backend in self.levels:
            task = backend.get_non_blocking(key, location)
            if task:
                return task
        return None

    # ------------------------------------------------------------------
    # put  (batch-preserving)
    # ------------------------------------------------------------------

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> Union[List[Future], None]:
        """
        Write to backends based on write policy.
        Batch is forwarded to each level as-is.
        """
        if self._write_policy == "write_back":
            _, backend = self.levels[0]
            return backend.batched_submit_put_task(keys, objs, transfer_spec)

        all_futures: List[Future] = []
        for _, backend in self.levels:
            futures = backend.batched_submit_put_task(
                keys, objs, transfer_spec,
            )
            if futures:
                all_futures.extend(futures)
        return all_futures if all_futures else None

    # ------------------------------------------------------------------
    # pin / unpin / touch
    # ------------------------------------------------------------------

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin the key at every level that contains it."""
        result = False
        for level_idx, (_, backend) in enumerate(self.levels):
            if backend.contains(key):
                backend.pin(key)
                with self._pin_registry_lock:
                    self._pin_registry.setdefault(key, set()).add(level_idx)
                result = True
        return result

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin the key only at levels where it was actually pinned."""
        with self._pin_registry_lock:
            pinned_levels = self._pin_registry.pop(key, None)
        if not pinned_levels:
            return False
        result = False
        for level_idx in pinned_levels:
            _, backend = self.levels[level_idx]
            backend.unpin(key)
            result = True
        return result

    def touch_cache(self):
        for _, backend in self.levels:
            if hasattr(backend, "touch_cache"):
                backend.touch_cache()

    # ------------------------------------------------------------------
    # remove / clear
    # ------------------------------------------------------------------

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        removed = False
        for _, backend in self.levels:
            try:
                if backend.remove(key, force):
                    removed = True
            except NotImplementedError:
                pass
        return removed

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        total_removed = 0
        for _, backend in self.levels:
            try:
                total_removed += backend.batched_remove(keys, force)
            except NotImplementedError:
                pass
        return total_removed

    def clear(self) -> int:
        total_cleared = 0
        for _, backend in self.levels:
            if hasattr(backend, "clear"):
                total_cleared += backend.clear()
        return total_cleared

    # ------------------------------------------------------------------
    # allocate (delegate to allocator backend)
    # ------------------------------------------------------------------

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> MemoryAllocatorInterface:
        assert self._allocator_backend is not None
        return self._allocator_backend.initialize_allocator(config, metadata)

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        assert self._allocator_backend is not None
        return self._allocator_backend.get_memory_allocator()

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        assert self._allocator_backend is not None
        return self._allocator_backend.allocate(
            shapes, dtypes, fmt, eviction=eviction, busy_loop=busy_loop,
        )

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        assert self._allocator_backend is not None
        return self._allocator_backend.batched_allocate(
            shapes, dtypes, batch_size, fmt, eviction=eviction, busy_loop=busy_loop,
        )

    def calculate_chunk_budget(self) -> int:
        assert self._allocator_backend is not None
        return self._allocator_backend.calculate_chunk_budget()

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        assert self._allocator_backend is not None
        return self._allocator_backend

    # ------------------------------------------------------------------
    # promotion  —  CPU level only
    # ------------------------------------------------------------------

    def _promote_to_cpu(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        """
        Promote a batch of cache entries to the CPU level (level 0).

        Only keys that are not already present in the CPU backend are
        promoted.  The sub-backend's ``batched_submit_put_task`` handles
        ref_count_up internally.
        """
        cpu_backend = self._cpu_level_backend

        to_promote_keys: List[CacheEngineKey] = []
        to_promote_objs: List[MemoryObj] = []
        for key, obj in zip(keys, memory_objs, strict=False):
            if not cpu_backend.contains(key):
                to_promote_keys.append(key)
                to_promote_objs.append(obj)

        if not to_promote_keys:
            return

        try:
            cpu_backend.batched_submit_put_task(to_promote_keys, to_promote_objs)
            logger.debug(
                "Promoted %d keys to CPU level (%s)",
                len(to_promote_keys),
                self._cpu_level_name,
            )
        except Exception as e:
            logger.warning(
                "Failed to promote %d keys to CPU level (%s): %s",
                len(to_promote_keys),
                self._cpu_level_name,
                e,
            )

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._get_pool.shutdown(wait=False)
        for name, backend in self.levels:
            try:
                backend.close()
                logger.info("XIOBackend: closed sub-backend %s", name)
            except Exception as e:
                logger.error(
                    "XIOBackend: error closing sub-backend %s: %s", name, e,
                )
