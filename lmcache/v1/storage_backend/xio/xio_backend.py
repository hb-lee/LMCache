# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio

# Third Party

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.xio.xio_config import XIOConfig

if TYPE_CHECKING:
    from lmcache.v1.memory_management import MemoryAllocatorInterface

logger = init_logger(__name__)


class XIOStorageBackend(StorageBackendInterface):
    """
    XIO (eXternal I/O) storage backend wrapper.

    This backend wraps an existing storage backend and provides parallel I/O
    capabilities by splitting large MemoryObj into smaller chunks and executing
    I/O operations concurrently using a thread pool.

    Optimization strategies to minimize memory copy:
    1. Use memoryview for zero-copy slicing
    2. Pass offset/length info to avoid data splitting
    3. Reuse original buffer references
    """

    def __init__(
        self,
        backend: StorageBackendInterface,
        config: XIOConfig,
        dst_device: str = "cuda",
    ):
        """
        Initialize the XIO storage backend.
        """
        super().__init__(dst_device)
        self.backend = backend
        self.config = config
        self._thread_pool: Optional[ThreadPoolExecutor] = None

        if self.config.enabled:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.config.thread_pool_size
            )
            logger.info(
                "XIOStorageBackend initialized with chunk_size=%d, "
                "thread_pool_size=%d",
                self.config.chunk_size,
                self.config.thread_pool_size,
            )
        else:
            logger.info("XIOStorageBackend initialized (disabled)")

    def _get_sub_key(self, key: CacheEngineKey, chunk_idx: int) -> CacheEngineKey:
        """
        Generate a sub-key for a chunk.
        """
        return CacheEngineKey(
            model_name=key.model_name,
            world_size=key.world_size,
            worker_id=key.worker_id,
            chunk_hash=key.chunk_hash,
            dtype=key.dtype,
            request_configs=key.request_configs,
        )

    def _calculate_chunks(
        self, memory_obj: MemoryObj
    ) -> List[tuple[int, int, int]]:
        """
        Calculate chunk boundaries for splitting a MemoryObj.
        
        Returns:
            List of (offset, length, chunk_idx) tuples.
        """
        total_size = memory_obj.get_physical_size()
        chunk_size = total_size // self.config.chunk_size

        chunks = []
        for i in range(self.config.chunk_size):
            offset = i * chunk_size
            if i == self.config.chunk_size - 1:
                length = total_size - offset
            else:
                length = chunk_size
            chunks.append((offset, length, i))

        return chunks

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether the key exists in the storage backend.
        """
        return self.backend.contains(key, pin=pin)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing put tasks.
        """
        return self.backend.exists_in_put_tasks(key)

    def _extract_chunk(
        self, memory_obj: MemoryObj, offset: int, length: int
    ) -> bytes:
        """
        Extract a chunk from MemoryObj with minimal copy.
        
        Uses memoryview to avoid copy when possible, then converts only
        the needed portion.
        """
        byte_array = memory_obj.byte_array
        if isinstance(byte_array, (memoryview, bytearray)):
            mv = memoryview(byte_array)
            chunk_view = mv[offset : offset + length]
            return bytes(chunk_view)
        return byte_array[offset : offset + length]

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[List[Future]]:
        """
        Submit batched put tasks with optional parallelization via XIO.
        
        When XIO is enabled, each MemoryObj is split into chunks and written
        in parallel. Each chunk uses minimal-copy extraction via memoryview.
        """
        if not self.config.enabled or not self.config.enable_put:
            return self.backend.batched_submit_put_task(
                keys, objs, transfer_spec, on_complete_callback
            )

        if self._thread_pool is None:
            return self.backend.batched_submit_put_task(
                keys, objs, transfer_spec, on_complete_callback
            )

        all_futures: List[Future] = []

        for key, memory_obj in zip(keys, objs, strict=False):
            chunks = self._calculate_chunks(memory_obj)

            futures = []
            for offset, length, chunk_idx in chunks:
                sub_key = self._get_sub_key(key, chunk_idx)
                
                chunk_bytes = self._extract_chunk(memory_obj, offset, length)

                future = self._thread_pool.submit(
                    self._write_chunk_to_backend,
                    sub_key,
                    chunk_bytes,
                    memory_obj.meta,
                )
                futures.append(future)

            all_futures.extend(futures)

        return all_futures if all_futures else None

    def _write_chunk_to_backend(
        self,
        key: CacheEngineKey,
        chunk_bytes: bytes,
        meta: Any,
    ) -> None:
        """Write a chunk to the underlying backend."""
        
        class ChunkWrapper:
            """Minimal wrapper to pass bytes as MemoryObj-like."""
            def __init__(self, data: bytes, m):
                self._data = data
                self._meta = m
            
            @property
            def byte_array(self) -> bytes:
                return self._data
            
            @property
            def tensor(self):
                return None
            
            @property
            def meta(self):
                return self._meta
            
            @property
            def metadata(self):
                return self._meta
            
            def get_physical_size(self) -> int:
                return len(self._data)

        wrapper = ChunkWrapper(chunk_bytes, meta)
        
        self.backend.batched_submit_put_task([key], [wrapper])

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Async version of batched_submit_put_task.
        """
        if not self.config.enabled or not self.config.enable_put:
            await self.backend.async_batched_submit_put_task(
                keys, objs, transfer_spec, on_complete_callback
            )
            return

        self.batched_submit_put_task(keys, objs, transfer_spec, on_complete_callback)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Get a MemoryObj from the storage backend.
        """
        if not self.config.enabled or not self.config.enable_get:
            return self.backend.get_blocking(key)

        return self.backend.get_blocking(key)

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        Non-blocking get.
        """
        return self.backend.get_non_blocking(key, location)

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check whether keys exist in the storage backend.
        """
        return await self.backend.batched_async_contains(lookup_id, keys, pin=pin)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        Non-blocking batched get.
        """
        return await self.backend.batched_get_non_blocking(lookup_id, keys, transfer_spec)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        Blocking batched get.

        If XIO is enabled, multiple keys are fetched in parallel using thread pool.
        """
        if not self.config.enabled or not self.config.enable_get:
            return self.backend.batched_get_blocking(keys)

        if self._thread_pool is None:
            return self.backend.batched_get_blocking(keys)

        results: List[Optional[MemoryObj]] = []

        futures = []
        for key in keys:
            future = self._thread_pool.submit(self.backend.get_blocking, key)
            futures.append(future)

        for future in futures:
            results.append(future.result())

        return results

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Pin a memory object.
        """
        return self.backend.pin(key)

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Unpin a memory object.
        """
        return self.backend.unpin(key)

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove a memory object.
        """
        return self.backend.remove(key, force=force)

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        force: bool = True,
    ) -> int:
        """
        Remove multiple memory objects.
        """
        return self.backend.batched_remove(keys, force=force)

    def get_allocator_backend(self) -> "AllocatorBackendInterface":
        """
        Get the allocator backend.
        """
        return self.backend.get_allocator_backend()

    def close(self) -> None:
        """
        Close the storage backend and thread pool.
        """
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

        self.backend.close()

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check whether keys exist in the storage backend.
        """
        return self.backend.batched_contains(keys, pin=pin)

    def __str__(self) -> str:
        return f"XIOStorageBackend({self.backend})"
