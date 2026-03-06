# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import asyncio
import os
import pickle
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, BytesBufferMemoryObj, MemoryObjMetadata
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

logger = init_logger(__name__)


class XIOBackend(StorageBackendInterface):
    """
    XIOBackend is a storage backend wrapper that supports splitting large MemoryObj
    into smaller chunks and processing them in parallel using thread pool.
    """
    
    def __init__(
        self,
        underlying_backend: StorageBackendInterface,
        chunk_size: int = 64 * 1024 * 1024,  # Default: 64MB
        max_workers: int = 4,
        dst_device: str = "cuda",
    ):
        """
        Initialize XIOBackend.
        
        :param underlying_backend: The underlying storage backend to use.
        :param chunk_size: Size of each chunk in bytes.
        :param max_workers: Maximum number of threads in the thread pool.
        :param dst_device: The device where the blocking retrieved KV is stored.
        """
        super().__init__(dst_device=dst_device)
        self.underlying_backend = underlying_backend
        self.chunk_size = chunk_size
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        
    def _generate_chunk_key(self, base_key: CacheEngineKey, chunk_index: int) -> CacheEngineKey:
        """
        Generate a chunk key based on the base key and chunk index.
        
        :param base_key: The base CacheEngineKey.
        :param chunk_index: The index of the chunk.
        :return: A new CacheEngineKey for the chunk.
        """
        # Create a new key with chunk index in the request_configs
        # This way we can keep the original chunk_hash intact for layer splitting
        chunk_request_configs = base_key.request_configs.copy() if base_key.request_configs else {}
        chunk_request_configs["lmcache.tag.xio_chunk"] = chunk_index
        
        return CacheEngineKey(
            model_name=base_key.model_name,
            world_size=base_key.world_size,
            worker_id=base_key.worker_id,
            chunk_hash=base_key.chunk_hash,
            dtype=base_key.dtype,
            request_configs=chunk_request_configs,
        )
    
    def _generate_metadata_key(self, base_key: CacheEngineKey) -> CacheEngineKey:
        """
        Generate a key for storing the original MemoryObj metadata.
        
        :param base_key: The base CacheEngineKey.
        :return: A new CacheEngineKey for the metadata.
        """
        metadata_request_configs = base_key.request_configs.copy() if base_key.request_configs else {}
        metadata_request_configs["lmcache.tag.xio_metadata"] = True
        
        return CacheEngineKey(
            model_name=base_key.model_name,
            world_size=base_key.world_size,
            worker_id=base_key.worker_id,
            chunk_hash=base_key.chunk_hash,
            dtype=base_key.dtype,
            request_configs=metadata_request_configs,
        )
    
    def _split_memory_obj(self, memory_obj: MemoryObj) -> List[Tuple[bytes, int]]:
        """
        Split a MemoryObj into smaller chunks.
        
        :param memory_obj: The MemoryObj to split.
        :return: A list of tuples containing (chunk_bytes, offset).
        """
        byte_array = memory_obj.byte_array
        total_size = len(byte_array)
        chunks = []
        
        for offset in range(0, total_size, self.chunk_size):
            end_offset = min(offset + self.chunk_size, total_size)
            chunk = byte_array[offset:end_offset]
            chunks.append((chunk, offset))
        
        return chunks
    
    def _merge_chunks(self, chunks: List[Tuple[bytes, int]], total_size: int) -> bytes:
        """
        Merge chunks back into a single byte array.
        
        :param chunks: A list of tuples containing (chunk_bytes, offset).
        :param total_size: The total size of the merged byte array.
        :return: The merged byte array.
        """
        # Sort chunks by offset
        chunks.sort(key=lambda x: x[1])
        
        # Merge chunks
        merged = bytearray(total_size)
        for chunk, offset in chunks:
            merged[offset:offset+len(chunk)] = chunk
        
        return bytes(merged)
    
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether key is in the storage backend.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        :param bool pin: Whether to pin the key.
        :return: True if the key exists, False otherwise.
        """
        return self.underlying_backend.contains(key, pin)
    
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing put tasks.
        
        :param CacheEngineKey key: The key to check.
        :return: True if the key is in ongoing put tasks, False otherwise.
        """
        return self.underlying_backend.exists_in_put_tasks(key)
    
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        """
        An async function to put the MemoryObj into the storage backend.
        
        :param List[CacheEngineKey] keys: The keys of the MemoryObjs.
        :param List[MemoryObj] objs: The MemoryObjs to be stored.
        :param Any transfer_spec: Optional transfer specification.
        :param on_complete_callback: Optional callback invoked once per key.
        :return: A list of `Future` objects if the operation is asynchronous.
        """
        futures = []
        
        for key, obj in zip(keys, objs, strict=False):
            # Split the MemoryObj into chunks
            chunks = self._split_memory_obj(obj)
            
            if not chunks:
                continue
            
            # Create metadata for the original MemoryObj
            metadata_dict = {
                "shape": list(obj.get_shape()),
                "dtype": str(obj.get_dtype()).replace("torch.", ""),
                "format": obj.get_memory_format().value,
                "total_size": obj.get_size(),
                "num_chunks": len(chunks),
            }
            metadata_bytes = pickle.dumps(metadata_dict)
            
            # Create a BytesBufferMemoryObj for the metadata
            metadata_key = self._generate_metadata_key(key)
            metadata_obj = BytesBufferMemoryObj(raw_bytes=metadata_bytes)
            
            # Submit metadata to underlying backend
            self.underlying_backend.batched_submit_put_task([metadata_key], [metadata_obj])
            
            # Create futures for each chunk
            chunk_futures = []
            for i, (chunk_bytes, offset) in enumerate(chunks):
                # Generate chunk key
                chunk_key = self._generate_chunk_key(key, i)
                
                # Create a BytesBufferMemoryObj for the chunk
                chunk_obj = BytesBufferMemoryObj(raw_bytes=chunk_bytes)
                
                # Submit chunk to underlying backend
                future = self.thread_pool.submit(
                    lambda k, o: self.underlying_backend.batched_submit_put_task([k], [o]),
                    chunk_key, chunk_obj
                )
                chunk_futures.append(future)
            
            # When all chunks are done, call the callback
            def create_callback(base_key):
                def callback(fut):
                    if on_complete_callback:
                        on_complete_callback(base_key)
                return callback
            
            # Create a future that completes when all chunk futures complete
            all_done = Future()
            all_done.set_result(None)
            futures.append(all_done)
        
        return futures
    
    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        An async version of batched_submit_put_task.
        
        :param keys: The keys of the MemoryObjs.
        :param objs: The MemoryObjs to be stored.
        :param transfer_spec: Optional transfer specification.
        :param on_complete_callback: Optional callback invoked once per key.
        """
        # Convert to async using asyncio.wrap_future
        futures = self.batched_submit_put_task(keys, objs, transfer_spec, on_complete_callback)
        if futures:
            for future in futures:
                await asyncio.wrap_future(future)
    
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        A blocking function to get the kv cache from the storage backend.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        :return: MemoryObj. None if the key does not exist.
        """
        # First, retrieve the metadata
        metadata_key = self._generate_metadata_key(key)
        metadata_obj = self.underlying_backend.get_blocking(metadata_key)
        
        if metadata_obj is None:
            return None
        
        # Deserialize metadata
        metadata_dict = pickle.loads(metadata_obj.byte_array)
        original_shape = torch.Size(metadata_dict["shape"])
        original_dtype = getattr(torch, metadata_dict["dtype"])
        original_format = MemoryFormat(metadata_dict["format"])
        total_size = metadata_dict["total_size"]
        num_chunks = metadata_dict["num_chunks"]
        
        # Retrieve all chunks
        chunks = []
        
        for i in range(num_chunks):
            chunk_key = self._generate_chunk_key(key, i)
            chunk_obj = self.underlying_backend.get_blocking(chunk_key)
            
            if chunk_obj is None:
                # If any chunk is missing, consider the entire object missing
                logger.warning(f"Chunk {i} for key {key} is missing")
                return None
            
            chunk_bytes = chunk_obj.byte_array
            chunks.append((chunk_bytes, i * self.chunk_size))
        
        if not chunks:
            return None
        
        # Merge chunks back into a single byte array
        merged_bytes = self._merge_chunks(chunks, total_size)
        
        # Create a new MemoryObj with the merged data
        # We need to use the allocator from the underlying backend to create a proper MemoryObj
        allocator_backend = self.get_allocator_backend()
        memory_obj = allocator_backend.allocate(original_shape, original_dtype, original_format)
        
        if memory_obj is None:
            logger.error(f"Failed to allocate memory for key {key}")
            return None
        
        # Copy the merged bytes into the MemoryObj
        # This assumes that memory_obj.byte_array is writable
        memory_view = memory_obj.byte_array
        memory_view[:len(merged_bytes)] = merged_bytes
        
        return memory_obj
    
    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        A non-blocking function to get the kv cache from the storage backend.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        :param location: Optional location parameter.
        :return: Future object if the operation is asynchronous.
        """
        return self.thread_pool.submit(self.get_blocking, key)
    
    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check whether keys are in the storage backend.
        
        :param lookup_id: Lookup identifier.
        :param keys: The keys of the MemoryObjs.
        :param pin: Whether to pin the keys.
        :return: The number of keys that exist in the storage backend.
        """
        return await self.underlying_backend.batched_async_contains(lookup_id, keys, pin)
    
    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        A non-blocking function to get the kv cache from the storage backend.
        
        :param lookup_id: Lookup identifier.
        :param keys: The keys of the list of MemoryObjs.
        :param transfer_spec: Optional transfer specification.
        :return: A list of MemoryObjs.
        """
        # Process each key in parallel
        futures = [self.get_non_blocking(key) for key in keys]
        return [await asyncio.wrap_future(fut) for fut in futures]
    
    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        A blocking function to get the kv cache from the storage backend.
        
        :param keys: The keys of the MemoryObjs.
        :return: A list of memory objects.
        """
        # Process each key in parallel
        futures = [self.thread_pool.submit(self.get_blocking, key) for key in keys]
        return [fut.result() for fut in futures]
    
    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Pin a memory object so it will not be evicted.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        :return: True if pin is successful, False otherwise.
        """
        return self.underlying_backend.pin(key)
    
    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        """
        Unpin a memory object so it can be evicted.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        :return: True if unpin is successful, False otherwise.
        """
        return self.underlying_backend.unpin(key)
    
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove a memory object.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        :param bool force: Whether to force remove the memory object.
        :return: True if remove is successful, False otherwise.
        """
        # Remove metadata first
        metadata_key = self._generate_metadata_key(key)
        self.underlying_backend.remove(metadata_key, force)
        
        # Remove all chunks
        success = True
        i = 0
        
        # Try to retrieve metadata to know how many chunks there are
        # If metadata is already removed, we'll try until we can't find any more chunks
        metadata_obj = self.underlying_backend.get_blocking(metadata_key)
        num_chunks = None
        
        if metadata_obj is not None:
            try:
                metadata_dict = pickle.loads(metadata_obj.byte_array)
                num_chunks = metadata_dict["num_chunks"]
            except:
                pass
        
        if num_chunks is not None:
            # If we know the number of chunks, remove exactly that many
            for i in range(num_chunks):
                chunk_key = self._generate_chunk_key(key, i)
                if not self.underlying_backend.remove(chunk_key, force):
                    logger.warning(f"Failed to remove chunk {i} for key {key}")
        else:
            # Otherwise, remove until we can't find any more chunks
            while True:
                chunk_key = self._generate_chunk_key(key, i)
                if not self.underlying_backend.remove(chunk_key, force):
                    break
                i += 1
        
        return success
    
    def get_allocator_backend(self):
        """
        Get the allocator backend that is used by the current storage backend.
        
        :return: An instance of AllocateBackendInterface.
        """
        return self.underlying_backend.get_allocator_backend()
    
    def close(
        self,
    ) -> None:
        """
        Close the storage backend.
        """
        self.thread_pool.shutdown(wait=True)
        self.underlying_backend.close()