# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.xio.config import XIOConfig
from lmcache.v1.storage_backend.xio.splitter import XIOSplitter, XIOSlice

if TYPE_CHECKING:
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class XIOBackend(StorageBackendInterface):
    """
    A storage backend wrapper that enables parallel I/O operations.

    XIO (eXecutable IO) splits large I/O operations into multiple smaller
    concurrent I/O operations to improve throughput for large KV cache transfers.

    This backend wraps an existing storage backend and transparently provides
    parallel I/O capabilities without requiring changes to the underlying backend.

    Attributes:
        underlying_backend: The storage backend to wrap.
        config: XIO configuration.
    """

    def __init__(
        self,
        underlying_backend: StorageBackendInterface,
        config: XIOConfig,
        dst_device: str = "cuda",
    ):
        """
        Initialize the XIO backend.

        Args:
            underlying_backend: The storage backend to wrap.
            config: XIO configuration.
            dst_device: The target device for tensor operations.
        """
        super().__init__(dst_device)

        self.underlying_backend = underlying_backend
        self.config = config

        if config.enabled:
            self.splitter = XIOSplitter(config.chunk_size)
            self.executor = ThreadPoolExecutor(max_workers=config.concurrency)
            logger.info(
                f"XIO enabled: chunk_size={config.chunk_size}, "
                f"concurrency={config.concurrency}"
            )
        else:
            self.splitter = None
            self.executor = None
            logger.info("XIO disabled, using passthrough mode")

        self.put_locks: dict[CacheEngineKey, threading.Lock] = {}
        self.locks_lock = threading.Lock()

    def _get_key_lock(self, key: CacheEngineKey) -> threading.Lock:
        """Get or create a lock for a specific key."""
        with self.locks_lock:
            if key not in self.put_locks:
                self.put_locks[key] = threading.Lock()
            return self.put_locks[key]

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists in the underlying backend."""
        return self.underlying_backend.contains(key, pin)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check if key exists in ongoing put tasks."""
        return self.underlying_backend.exists_in_put_tasks(key)

    def _get_slices(self, memory_obj: MemoryObj) -> List[XIOSlice]:
        """Get the slices for a memory object."""
        if self.splitter is None:
            return [XIOSlice(index=0, offset=0, size=memory_obj.get_physical_size())]
        return self.splitter.split(memory_obj.get_physical_size())

    def _build_chunk_key(self, key: CacheEngineKey, slice_info: XIOSlice) -> str:
        """Build a unique key for each chunk."""
        return f"{key.to_string()}_slice_{slice_info.index}"

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        """
        Submit batched put tasks with parallel I/O.

        If XIO is enabled, splits each MemoryObj into chunks and writes them
        in parallel using a thread pool.
        """
        if not self.config.enabled:
            return self.underlying_backend.batched_submit_put_task(
                keys, objs, transfer_spec, on_complete_callback
            )

        for key, memory_obj in zip(keys, objs, strict=False):
            self._put_with_xio(key, memory_obj, on_complete_callback)

    def _put_with_xio(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        """
        Put a MemoryObj with parallel I/O.

        Splits the MemoryObj into slices and writes them concurrently.
        """
        slices = self._get_slices(memory_obj)

        if len(slices) == 1:
            self.underlying_backend.batched_submit_put_task(
                [key], [memory_obj], None, on_complete_callback
            )
            return

        memory_obj.ref_count_up()

        def write_slice(slice_info: XIOSlice):
            chunk_key = self._build_chunk_key(key, slice_info)
            chunk_data = memory_obj.byte_array[
                slice_info.offset : slice_info.offset + slice_info.size
            ]
            buffer = bytearray(chunk_data)

            temp_obj = _SliceMemoryObj(
                data=buffer,
                original_key=key,
                slice_info=slice_info,
            )

            self.underlying_backend.batched_submit_put_task(
                [key], [temp_obj], None, None
            )

        futures = []
        for slice_info in slices:
            future = self.executor.submit(write_slice, slice_info)
            futures.append(future)

        def wait_and_complete():
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to write slice for {key}: {e}")

            memory_obj.ref_count_down()

            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(
                        f"on_complete_callback failed for key {key}: {e}"
                    )

        self.executor.submit(wait_and_complete)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Blocking get function.

        If XIO is enabled, reads chunks in parallel and assembles them.
        """
        if not self.config.enabled:
            return self.underlying_backend.get_blocking(key)

        memory_obj = self.underlying_backend.get_blocking(key)
        if memory_obj is None:
            return None

        return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        Non-blocking batched get with parallel I/O.

        If XIO is enabled, reads chunks in parallel and assembles them.
        """
        if not self.config.enabled:
            return await self.underlying_backend.batched_get_non_blocking(
                lookup_id, keys, transfer_spec
            )

        results = await self.underlying_backend.batched_get_non_blocking(
            lookup_id, keys, transfer_spec
        )
        return results

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a memory object in the underlying backend."""
        return self.underlying_backend.pin(key)

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a memory object in the underlying backend."""
        return self.underlying_backend.unpin(key)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a memory object from the underlying backend."""
        return self.underlying_backend.remove(key, force)

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        """Get the allocator backend from the underlying backend."""
        return self.underlying_backend.get_allocator_backend()

    def close(self):
        """Close the XIO backend and underlying backend."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        self.underlying_backend.close()

    def __str__(self) -> str:
        return f"XIOBackend({self.underlying_backend})"


class _SliceMemoryObj(MemoryObj):
    """
    A wrapper MemoryObj for a slice of data.

    This is used internally by XIO to represent a chunk of data
    that needs to be stored/retrieved separately.
    """

    def __init__(self, data: bytes, original_key: CacheEngineKey, slice_info: XIOSlice):
        from lmcache.v1.memory_management import MemoryObjMetadata

        self._data = data
        self._original_key = original_key
        self._slice_info = slice_info

        metadata = MemoryObjMetadata(
            shape=torch.Size([len(data)]),
            dtype=torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
            cached_positions=None,
        )
        super().__init__(metadata)

    def invalidate(self):
        pass

    def is_valid(self) -> bool:
        return True

    def get_size(self) -> int:
        return len(self._data)

    def get_shape(self) -> torch.Size:
        return torch.Size([len(self._data)])

    def get_dtype(self) -> torch.dtype:
        return torch.uint8

    def get_shapes(self) -> list[torch.Size]:
        return [torch.Size([len(self._data)])]

    def get_dtypes(self) -> list[torch.dtype]:
        return [torch.uint8]

    def get_memory_format(self) -> MemoryFormat:
        return MemoryFormat.KV_2LTD

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
    def tensor(self) -> Optional[torch.Tensor]:
        return torch.frombuffer(self._data, dtype=torch.uint8)

    @property
    def byte_array(self) -> bytes:
        return self._data

    @property
    def data_ptr(self) -> int:
        return 0
