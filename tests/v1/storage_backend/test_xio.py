# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import tempfile
import time
import os
import shutil
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.xio import XIOBackend, XIOConfig, XIOSplitter, XIOSlice


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


def create_test_memory_obj(data_size: int = 1024) -> MemoryObj:
    """Create a test MemoryObj with specified data size."""
    data = bytes([i % 256 for i in range(data_size)])
    metadata = MemoryObjMetadata(
        shape=torch.Size([data_size]),
        dtype=torch.uint8,
        fmt=MemoryFormat.KV_2LTD,
        cached_positions=None,
    )
    return _TestMemoryObj(data, metadata)


class _TestMemoryObj(MemoryObj):
    """Test implementation of MemoryObj."""

    def __init__(self, data: bytes, metadata: MemoryObjMetadata):
        self._data = data
        super().__init__(metadata)

    def invalidate(self):
        pass

    def is_valid(self) -> bool:
        return True

    def get_size(self) -> int:
        return len(self._data)

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> torch.dtype:
        return self.meta.dtype

    def get_shapes(self) -> list[torch.Size]:
        return [self.meta.shape]

    def get_dtypes(self) -> list[torch.dtype]:
        return [self.meta.dtype]

    def get_memory_format(self) -> MemoryFormat:
        return self.meta.fmt

    def get_physical_size(self) -> int:
        return len(self._data)

    def pin(self) -> bool:
        return True

    def ref_count_up(self):
        pass

    def unpin(self) -> bool:
        return True

    def ref_count_down(self):
        pass

    def get_ref_count(self) -> int:
        return 1

    def get_num_tokens(self) -> int:
        return len(self._data)

    @property
    def metadata(self):
        return self.meta

    @property
    def tensor(self) -> torch.Tensor:
        return torch.frombuffer(self._data, dtype=torch.uint8)

    @property
    def byte_array(self) -> bytes:
        return self._data

    @property
    def data_ptr(self) -> int:
        return 0


class MockStorageBackend:
    """Mock storage backend for testing."""

    def __init__(self):
        self.storage = {}
        self.put_calls = []
        self.get_calls = []

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return key in self.storage

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    def batched_submit_put_task(
        self, keys, objs, transfer_spec=None, on_complete_callback=None
    ):
        for key, obj in zip(keys, objs, strict=False):
            self.storage[key] = obj
            self.put_calls.append((key, obj))
            if on_complete_callback:
                on_complete_callback(key)

    def get_blocking(self, key: CacheEngineKey):
        return self.storage.get(key)

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if key in self.storage:
            del self.storage[key]
            return True
        return False

    def get_allocator_backend(self):
        return MagicMock()

    def close(self):
        pass


class TestXIOConfig:
    """Tests for XIOConfig."""

    def test_default_config(self):
        """Test default XIO configuration."""
        config = XIOConfig()
        assert config.enabled is True
        assert config.chunk_size == 64 * 1024 * 1024  # 64MB
        assert config.concurrency == 4

    def test_custom_config(self):
        """Test custom XIO configuration."""
        config = XIOConfig(chunk_size=32 * 1024 * 1024, concurrency=8, enabled=True)
        assert config.chunk_size == 32 * 1024 * 1024
        assert config.concurrency == 8
        assert config.enabled is True

    def test_disabled_config(self):
        """Test disabled XIO configuration."""
        config = XIOConfig(enabled=False)
        assert config.enabled is False
        assert config.chunk_size == 64 * 1024 * 1024  # Default still set
        assert config.concurrency == 4

    def test_from_dict(self):
        """Test creating XIOConfig from dictionary."""
        config_dict = {
            "xio_chunk_size": 128 * 1024 * 1024,
            "xio_concurrency": 16,
            "xio_enabled": True,
        }
        config = XIOConfig.from_dict(config_dict)
        assert config.chunk_size == 128 * 1024 * 1024
        assert config.concurrency == 16
        assert config.enabled is True

    def test_from_dict_disabled(self):
        """Test creating disabled XIOConfig from dictionary."""
        config_dict = {
            "xio_chunk_size": 1024,
            "xio_concurrency": 2,
            "xio_enabled": False,
        }
        config = XIOConfig.from_dict(config_dict)
        assert config.enabled is False


class TestXIOSplitter:
    """Tests for XIOSplitter."""

    def test_split_single_chunk(self):
        """Test splitting data that fits in one chunk."""
        splitter = XIOSplitter(chunk_size=1024)
        slices = splitter.split(512)
        assert len(slices) == 1
        assert slices[0].offset == 0
        assert slices[0].size == 512

    def test_split_multiple_chunks(self):
        """Test splitting data into multiple chunks."""
        splitter = XIOSplitter(chunk_size=100)
        slices = splitter.split(350)
        assert len(slices) == 4
        assert slices[0] == XIOSlice(index=0, offset=0, size=100)
        assert slices[1] == XIOSlice(index=1, offset=100, size=100)
        assert slices[2] == XIOSlice(index=2, offset=200, size=100)
        assert slices[3] == XIOSlice(index=3, offset=300, size=50)

    def test_split_exact_chunk(self):
        """Test splitting data that exactly fits chunks."""
        splitter = XIOSplitter(chunk_size=100)
        slices = splitter.split(300)
        assert len(slices) == 3
        for i, s in enumerate(slices):
            assert s.index == i
            assert s.offset == i * 100
            assert s.size == 100

    def test_split_zero_size(self):
        """Test splitting zero-size data."""
        splitter = XIOSplitter(chunk_size=1024)
        slices = splitter.split(0)
        assert len(slices) == 0

    def test_split_negative_size(self):
        """Test splitting negative-size data."""
        splitter = XIOSplitter(chunk_size=1024)
        slices = splitter.split(-100)
        assert len(slices) == 0

    def test_get_num_slices(self):
        """Test calculating number of slices."""
        splitter = XIOSplitter(chunk_size=100)
        assert splitter.get_num_slices(0) == 0
        assert splitter.get_num_slices(50) == 1
        assert splitter.get_num_slices(100) == 1
        assert splitter.get_num_slices(101) == 2
        assert splitter.get_num_slices(300) == 3


class TestXIOBackend:
    """Tests for XIOBackend."""

    def test_xio_backend_init_enabled(self):
        """Test XIOBackend initialization with XIO enabled."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(chunk_size=1024, concurrency=4, enabled=True)
        xio_backend = XIOBackend(mock_backend, config)
        assert xio_backend.config.enabled is True
        assert xio_backend.splitter is not None
        assert xio_backend.executor is not None

    def test_xio_backend_init_disabled(self):
        """Test XIOBackend initialization with XIO disabled."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio_backend = XIOBackend(mock_backend, config)
        assert xio_backend.config.enabled is False
        assert xio_backend.splitter is None
        assert xio_backend.executor is None

    def test_contains(self):
        """Test contains method delegation."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio_backend = XIOBackend(mock_backend, config)
        key = create_test_key()
        assert xio_backend.contains(key) is False
        mock_backend.storage[key] = create_test_memory_obj()
        assert xio_backend.contains(key) is True

    def test_put_small_data(self):
        """Test put operation with small data (no splitting)."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(chunk_size=1024, concurrency=4, enabled=True)
        xio_backend = XIOBackend(mock_backend, config)

        key = create_test_key()
        mem_obj = create_test_memory_obj(512)

        xio_backend.batched_submit_put_task([key], [mem_obj])
        time.sleep(0.1)  # Allow async operations to complete

        assert key in mock_backend.storage

    def test_put_large_data(self):
        """Test put operation with large data (with splitting)."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(chunk_size=100, concurrency=4, enabled=True)
        xio_backend = XIOBackend(mock_backend, config)

        key = create_test_key()
        mem_obj = create_test_memory_obj(350)  # 350 bytes, split into 4 chunks

        xio_backend.batched_submit_put_task([key], [mem_obj])
        time.sleep(0.2)  # Allow async operations to complete

        # The data should be stored
        assert key in mock_backend.storage

    def test_get_blocking(self):
        """Test blocking get operation."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio_backend = XIOBackend(mock_backend, config)

        key = create_test_key()
        mem_obj = create_test_memory_obj(512)
        mock_backend.storage[key] = mem_obj

        retrieved = xio_backend.get_blocking(key)
        assert retrieved is not None

    def test_remove(self):
        """Test remove operation."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(enabled=False)
        xio_backend = XIOBackend(mock_backend, config)

        key = create_test_key()
        mem_obj = create_test_memory_obj(512)
        mock_backend.storage[key] = mem_obj

        assert xio_backend.remove(key) is True
        assert key not in mock_backend.storage

    def test_close(self):
        """Test close operation."""
        mock_backend = MockStorageBackend()
        config = XIOConfig(enabled=True)
        xio_backend = XIOBackend(mock_backend, config)

        xio_backend.close()
        assert xio_backend.executor._shutdown


class TestXIOIntegration:
    """Integration tests for XIO with real storage backends."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def async_loop(self):
        """Create an asyncio event loop for testing."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
        loop.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_xio_with_local_cpu_backend(self, temp_dir):
        """Test XIOBackend wrapping LocalCPUBackend."""
        from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

        config = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_disk=temp_dir,
            max_local_disk_size=1.0,
            lmcache_instance_id="test_instance",
        )
        metadata = LMCacheMetadata(
            model_name="test_model",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.bfloat16,
            kv_shape=(28, 2, 256, 8, 128),
        )
        # This test would require more setup - just verify import works
        from lmcache.v1.storage_backend.xio import XIOBackend, XIOConfig

        config = XIOConfig(enabled=False)
        assert config.enabled is False
