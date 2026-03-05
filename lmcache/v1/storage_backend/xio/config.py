# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional


@dataclass
class XIOConfig:
    """
    Configuration for XIO (eXecutable IO) module.

    XIO splits large I/O operations into multiple smaller concurrent I/O
    operations to improve throughput for large KV cache transfers.

    Attributes:
        chunk_size: The size in bytes to split each I/O into.
            If None or 0, no splitting will occur.
        concurrency: Maximum number of concurrent I/O operations.
            If None, defaults to 4.
        enabled: Whether XIO is enabled. If False, passthrough mode.
    """

    chunk_size: Optional[int] = None
    concurrency: Optional[int] = None
    enabled: bool = True

    def __post_init__(self):
        if self.enabled:
            if self.chunk_size is None or self.chunk_size <= 0:
                self.chunk_size = 64 * 1024 * 1024  # 64MB default
            if self.concurrency is None or self.concurrency <= 0:
                self.concurrency = 4

    @staticmethod
    def from_dict(config_dict: dict) -> "XIOConfig":
        """
        Create XIOConfig from a dictionary.

        Args:
            config_dict: Dictionary containing xio configuration.
                Expected keys: xio_chunk_size, xio_concurrency, xio_enabled

        Returns:
            XIOConfig instance
        """
        chunk_size = config_dict.get("xio_chunk_size")
        concurrency = config_dict.get("xio_concurrency")
        enabled = config_dict.get("xio_enabled", True)

        if chunk_size is not None:
            chunk_size = int(chunk_size)
        if concurrency is not None:
            concurrency = int(concurrency)
        enabled = bool(enabled)

        return XIOConfig(
            chunk_size=chunk_size,
            concurrency=concurrency,
            enabled=enabled,
        )
