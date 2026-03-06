# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, BytesBufferMemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.xio_backend import XIOBackend


class TestXIOBackend(unittest.TestCase):
    """
    Test cases for XIOBackend class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock underlying backend
        self.mock_backend = Mock(spec=StorageBackendInterface)
        
        # Create a mock allocator backend
        self.mock_allocator = Mock()
        self.mock_backend.get_allocator_backend.return_value = self.mock_allocator
        
        # Create XIOBackend instance
        self.xio_backend = XIOBackend(
            underlying_backend=self.mock_backend,
            chunk_size=1024,  # Small chunk size for testing
            max_workers=2
        )
        
        # Create a test CacheEngineKey
        self.test_key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=12345,
            dtype=torch.float32,
            request_configs={}
        )
        
    def test_generate_chunk_key(self):
        """
        Test that chunk keys are generated correctly.
        """
        chunk_key = self.xio_backend._generate_chunk_key(self.test_key, 0)
        
        # Check that the chunk key has the correct properties
        self.assertEqual(chunk_key.model_name, self.test_key.model_name)
        self.assertEqual(chunk_key.world_size, self.test_key.world_size)
        self.assertEqual(chunk_key.worker_id, self.test_key.worker_id)
        self.assertEqual(chunk_key.chunk_hash, self.test_key.chunk_hash)
        self.assertEqual(chunk_key.dtype, self.test_key.dtype)
        
        # Check that the chunk index is in the request_configs
        self.assertIn("lmcache.tag.xio_chunk", chunk_key.request_configs)
        self.assertEqual(chunk_key.request_configs["lmcache.tag.xio_chunk"], 0)
    
    def test_generate_metadata_key(self):
        """
        Test that metadata keys are generated correctly.
        """
        metadata_key = self.xio_backend._generate_metadata_key(self.test_key)
        
        # Check that the metadata key has the correct properties
        self.assertEqual(metadata_key.model_name, self.test_key.model_name)
        self.assertEqual(metadata_key.world_size, self.test_key.world_size)
        self.assertEqual(metadata_key.worker_id, self.test_key.worker_id)
        self.assertEqual(metadata_key.chunk_hash, self.test_key.chunk_hash)
        self.assertEqual(metadata_key.dtype, self.test_key.dtype)
        
        # Check that the metadata flag is in the request_configs
        self.assertIn("lmcache.tag.xio_metadata", metadata_key.request_configs)
        self.assertTrue(metadata_key.request_configs["lmcache.tag.xio_metadata"])
    
    def test_split_memory_obj(self):
        """
        Test that MemoryObj is split into chunks correctly.
        """
        # Create a test byte array
        test_data = b"x" * 3000  # 3000 bytes
        
        # Create a BytesBufferMemoryObj
        memory_obj = BytesBufferMemoryObj(raw_bytes=test_data)
        
        # Split the memory object
        chunks = self.xio_backend._split_memory_obj(memory_obj)
        
        # Check that we got the correct number of chunks
        # Chunk size is 1024, so 3000 / 1024 = 2.929, so 3 chunks
        self.assertEqual(len(chunks), 3)
        
        # Check chunk sizes
        self.assertEqual(len(chunks[0][0]), 1024)
        self.assertEqual(len(chunks[1][0]), 1024)
        self.assertEqual(len(chunks[2][0]), 3000 - 2048)
        
        # Check offsets
        self.assertEqual(chunks[0][1], 0)
        self.assertEqual(chunks[1][1], 1024)
        self.assertEqual(chunks[2][1], 2048)
    
    def test_merge_chunks(self):
        """
        Test that chunks are merged back correctly.
        """
        # Create test chunks
        chunks = [
            (b"abc", 0),
            (b"def", 3),
            (b"ghi", 6)
        ]
        
        # Merge chunks
        merged = self.xio_backend._merge_chunks(chunks, 9)
        
        # Check that the merged data is correct
        self.assertEqual(merged, b"abcdefghi")
    
    @patch("lmcache.v1.storage_backend.xio_backend.BytesBufferMemoryObj")
    def test_batched_submit_put_task(self, mock_bytes_buffer):
        """
        Test that batched_submit_put_task works correctly.
        """
        # Create test data
        test_data = b"x" * 3000
        
        # Create a mock MemoryObj
        mock_obj = Mock(spec=MemoryObj)
        mock_obj.byte_array = test_data
        mock_obj.get_shape.return_value = torch.Size([3000])
        mock_obj.get_dtype.return_value = torch.uint8
        mock_obj.get_memory_format.return_value = MemoryFormat.BINARY_BUFFER
        mock_obj.get_size.return_value = 3000
        
        # Create mock BytesBufferMemoryObj instances
        mock_metadata_buffer = Mock(spec=BytesBufferMemoryObj)
        mock_chunk_buffer_1 = Mock(spec=BytesBufferMemoryObj)
        mock_chunk_buffer_2 = Mock(spec=BytesBufferMemoryObj)
        mock_chunk_buffer_3 = Mock(spec=BytesBufferMemoryObj)
        mock_bytes_buffer.side_effect = [
            mock_metadata_buffer,  # For metadata
            mock_chunk_buffer_1,   # For chunk 0
            mock_chunk_buffer_2,   # For chunk 1
            mock_chunk_buffer_3    # For chunk 2
        ]
        
        # Call batched_submit_put_task
        self.xio_backend.batched_submit_put_task([self.test_key], [mock_obj])
        
        # Check that underlying_backend.batched_submit_put_task was called 4 times
        # (1 for metadata, 3 for chunks)
        self.assertEqual(self.mock_backend.batched_submit_put_task.call_count, 4)
    
    @patch("lmcache.v1.storage_backend.xio_backend.BytesBufferMemoryObj")
    def test_get_blocking(self, mock_bytes_buffer):
        """
        Test that get_blocking works correctly.
        """
        # Create test metadata
        metadata_dict = {
            "shape": [3000],
            "dtype": "uint8",
            "format": MemoryFormat.BINARY_BUFFER.value,
            "total_size": 3000,
            "num_chunks": 3
        }
        import pickle
        metadata_bytes = pickle.dumps(metadata_dict)
        
        # Create test chunks
        chunk1 = b"x" * 1024
        chunk2 = b"x" * 1024
        chunk3 = b"x" * (3000 - 2048)
        
        # Create mock BytesBufferMemoryObj instances
        mock_metadata_obj = Mock(spec=BytesBufferMemoryObj)
        mock_metadata_obj.byte_array = metadata_bytes
        
        mock_chunk_obj_1 = Mock(spec=BytesBufferMemoryObj)
        mock_chunk_obj_1.byte_array = chunk1
        
        mock_chunk_obj_2 = Mock(spec=BytesBufferMemoryObj)
        mock_chunk_obj_2.byte_array = chunk2
        
        mock_chunk_obj_3 = Mock(spec=BytesBufferMemoryObj)
        mock_chunk_obj_3.byte_array = chunk3
        
        # Set up mock_backend.get_blocking to return the mock objects
        def mock_get_blocking(key):
            if "lmcache.tag.xio_metadata" in key.request_configs:
                return mock_metadata_obj
            elif "lmcache.tag.xio_chunk" in key.request_configs:
                chunk_index = key.request_configs["lmcache.tag.xio_chunk"]
                if chunk_index == 0:
                    return mock_chunk_obj_1
                elif chunk_index == 1:
                    return mock_chunk_obj_2
                elif chunk_index == 2:
                    return mock_chunk_obj_3
            return None
        
        self.mock_backend.get_blocking.side_effect = mock_get_blocking
        
        # Create a mock memory object for the allocator
        mock_memory_obj = Mock(spec=MemoryObj)
        mock_memory_obj.byte_array = bytearray(3000)
        self.mock_allocator.allocate.return_value = mock_memory_obj
        
        # Call get_blocking
        result = self.xio_backend.get_blocking(self.test_key)
        
        # Check that the result is not None
        self.assertIsNotNone(result)
        
        # Check that the allocator was called with the correct parameters
        self.mock_allocator.allocate.assert_called_once_with(
            torch.Size([3000]),
            torch.uint8,
            MemoryFormat.BINARY_BUFFER
        )
    
    def test_remove(self):
        """
        Test that remove works correctly.
        """
        # Create test metadata
        metadata_dict = {
            "shape": [3000],
            "dtype": "uint8",
            "format": MemoryFormat.BINARY_BUFFER.value,
            "total_size": 3000,
            "num_chunks": 3
        }
        import pickle
        metadata_bytes = pickle.dumps(metadata_dict)
        
        # Create mock metadata object
        mock_metadata_obj = Mock(spec=BytesBufferMemoryObj)
        mock_metadata_obj.byte_array = metadata_bytes
        
        # Set up mock_backend.get_blocking to return the mock metadata object
        self.mock_backend.get_blocking.return_value = mock_metadata_obj
        
        # Call remove
        self.xio_backend.remove(self.test_key)
        
        # Check that underlying_backend.remove was called 4 times
        # (1 for metadata, 3 for chunks)
        self.assertEqual(self.mock_backend.remove.call_count, 4)


if __name__ == "__main__":
    unittest.main()