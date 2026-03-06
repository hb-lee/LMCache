# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional

# Third Party

# First Party


@dataclass
class XIOConfig:
    """
    Configuration for the XIO (eXternal I/O) module.

    The XIO module provides parallel I/O capabilities by splitting large
    MemoryObj into smaller chunks and executing I/O operations concurrently
    using a thread pool.
    """

    chunk_size: int = 4
    """
    The size of each chunk when splitting a MemoryObj for parallel I/O.
    This determines the number of chunks the data will be divided into.
    """

    thread_pool_size: int = 4
    """
    The number of worker threads in the thread pool for concurrent I/O.
    """

    enabled: bool = True
    """
    Whether XIO is enabled. When disabled, operations are forwarded
    directly to the underlying storage backend without splitting.
    """

    enable_get: bool = True
    """
    Whether to enable parallel get operations.
    """

    enable_put: bool = True
    """
    Whether to enable parallel put operations.
    """

    @staticmethod
    def from_dict(config: dict) -> "XIOConfig":
        """
        Create XIOConfig from a dictionary.

        Args:
            config: A dictionary containing XIO configuration.

        Returns:
            XIOConfig instance.
        """
        return XIOConfig(
            chunk_size=config.get("xio_chunk_size", 4),
            thread_pool_size=config.get("xio_thread_pool_size", 4),
            enabled=config.get("xio_enabled", True),
            enable_get=config.get("xio_enable_get", True),
            enable_put=config.get("xio_enable_put", True),
        )
