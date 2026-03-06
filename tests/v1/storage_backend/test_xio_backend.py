# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.xio.xio_backend import XIOStorageBackend
from lmcache.v1.storage_backend.xio.xio_config import XIOConfig

if TYPE_CHECKING:
    from lmcache.v1.memory_management import MemoryAllocatorInterface


class MockMemoryObj(MemoryObj):
    """Mock MemoryObj for testing."""

    def __init__(self, data: bytes, shape, dtype):
        metadata = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=0,
            phy_size=len(data),
            ref_count=0,
            fmt=MemoryFormat.KV_2LTD,
        )
        super().__init__(metadata)
        self._data = data

    def invalidate(self):
        pass

    def is_valid(self):
        return True

    def get_size(self) -> int:
        return len(self._data)

    def get_shape(self):
        return self.metadata.shape

    def get_dtype(self):
        return self.metadata.dtype

    def get_shapes(self):
        return [self.metadata.shape]

    def get_dtypes(self):
        return [self.metadata.dtype]

    def get_memory_format(self):
        return self.metadata.fmt

    def get_physical_size(self) -> int:
        return len(self._data)

    def pin(self) -> bool:
        return True

    def ref_count_up(self):
        self.metadata.ref_count += 1

    def unpin(self) -> bool:
        return True

    def ref_count_down(self):
        if self.metadata.ref_count > 0:
            self.metadata.ref_count -= 1

    def get_ref_count(self) -> int:
        return self.metadata.ref_count

    def get_num_tokens(self) -> int:
        return self.metadata.shape[2] if len(self.metadata.shape) >= 3 else 0

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def data_ptr(self) -> int:
        return 0

    @property
    def byte_array(self) -> bytes:
        return self._data

    @property
    def tensor(self):
        return None

    @property
    def is_pinned(self) -> bool:
        return False

    @property
    def can_evict(self) -> bool:
        return True

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        return None

    def get_tensor(self, index: int) -> Optional[torch.Tensor]:
        return None

    def parent(self) -> Optional["MemoryAllocatorInterface"]:
        return None


class MockStorageBackend(StorageBackendInterface):
    """Mock storage backend for testing."""

    def __init__(self):
        super().__init__(dst_device="cpu")
        self.data = {}
        self.put_tasks = set()
        self.lock = threading.Lock()

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.lock:
            return key in self.data

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List, None]:
        with self.lock:
            for key, obj in zip(keys, objs, strict=False):
                self.put_tasks.add(key)
                self.data[key] = obj
            return None

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        self.batched_submit_put_task(keys, objs, transfer_spec, on_complete_callback)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.lock:
            return self.data.get(key)

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ):
        return None

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        count = 0
        with self.lock:
            for key in keys:
                if key in self.data:
                    count += 1
                else:
                    break
        return count

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        results = []
        for key in keys:
            result = self.get_blocking(key)
            if result is not None:
                results.append(result)
            else:
                results.append(result)
        return results

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        return [self.get_blocking(key) for key in keys]

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        with self.lock:
            if key in self.data:
                del self.data[key]
                return True
            return False

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        count = 0
        for key in keys:
            if self.remove(key, force):
                count += 1
        return count

    def get_allocator_backend(self):
        return MagicMock()

    def close(self) -> None:
        pass

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        count = 0
        with self.lock:
            for key in keys:
                if key in self.data:
                    count += 1
                else:
                    break
        return count


def make_test_key(model_name: str = "test_model", chunk_hash: int = 12345) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name=model_name,
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
    )


class TestXIOConfig:
    """Tests for XIOConfig."""

    def test_default_config(self):
        config = XIOConfig()
        assert config.chunk_size == 4
        assert config.thread_pool_size == 4
        assert config.enabled is True
        assert config.enable_get is True
        assert config.enable_put is True

    def test_from_dict(self):
        config_dict = {
            "xio_chunk_size": 8,
            "xio_thread_pool_size": 16,
            "xio_enabled": False,
            "xio_enable_get": False,
            "xio_enable_put": True,
        }
        config = XIOConfig.from_dict(config_dict)
        assert config.chunk_size == 8
        assert config.thread_pool_size == 16
        assert config.enabled is False
        assert config.enable_get is False
        assert config.enable_put is True

    def test_from_dict_defaults(self):
        config = XIOConfig.from_dict({})
        assert config.chunk_size == 4
        assert config.thread_pool_size == 4
        assert config.enabled is True


class TestXIOStorageBackend:
    """Tests for XIOStorageBackend."""

    def test_init_enabled(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=True, chunk_size=4, thread_pool_size=4)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")
        assert xio._thread_pool is not None
        xio.close()

    def test_init_disabled(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")
        assert xio._thread_pool is None

    def test_contains(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=True)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")
        key = make_test_key()
        assert xio.contains(key) is False
        xio.close()

    def test_put_and_get(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")

        key = make_test_key()
        data = b"test_data" * 100
        shape = torch.Size([1, 2, 10])
        memory_obj = MockMemoryObj(data, shape, torch.float16)

        xio.batched_submit_put_task([key], [memory_obj])

        result = xio.get_blocking(key)
        assert result is not None

        xio.close()

    def test_batched_get_blocking(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")

        keys = [make_test_key(chunk_hash=i) for i in range(3)]
        for key in keys:
            data = b"test_data"
            shape = torch.Size([1, 2, 10])
            memory_obj = MockMemoryObj(data, shape, torch.float16)
            backend.batched_submit_put_task([key], [memory_obj])

        results = xio.batched_get_blocking(keys)
        assert len(results) == 3

        xio.close()

    def test_pin_unpin(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")

        key = make_test_key()
        assert xio.pin(key) is True
        assert xio.unpin(key) is True

        xio.close()

    def test_remove(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")

        key = make_test_key()
        data = b"test_data"
        shape = torch.Size([1, 2, 10])
        memory_obj = MockMemoryObj(data, shape, torch.float16)
        xio.batched_submit_put_task([key], [memory_obj])

        assert xio.remove(key) is True
        assert xio.contains(key) is False

        xio.close()

    def test_str(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")
        assert "XIOStorageBackend" in str(xio)
        xio.close()


class TestXIOStorageBackendWithXIOEnabled:
    """Tests for XIOStorageBackend with XIO enabled (parallel I/O)."""

    def test_batched_put_parallel(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=True, chunk_size=4, thread_pool_size=4)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")

        keys = [make_test_key(chunk_hash=i) for i in range(2)]
        for key in keys:
            data = b"test_data" * 100
            shape = torch.Size([1, 2, 10])
            memory_obj = MockMemoryObj(data, shape, torch.float16)
            xio.batched_submit_put_task([key], [memory_obj])

        xio.close()

    def test_batched_get_parallel(self):
        backend = MockStorageBackend()
        config = XIOConfig(enabled=True, chunk_size=4, thread_pool_size=4)
        xio = XIOStorageBackend(backend, config, dst_device="cpu")

        keys = [make_test_key(chunk_hash=i) for i in range(3)]
        for key in keys:
            data = b"test_data"
            shape = torch.Size([1, 2, 10])
            memory_obj = MockMemoryObj(data, shape, torch.float16)
            backend.batched_submit_put_task([key], [memory_obj])

        results = xio.batched_get_blocking(keys)
        assert len(results) == 3

        xio.close()
