# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class XIOSlice:
    """
    Represents a slice of a MemoryObj for parallel I/O operations.

    Attributes:
        index: The index of this slice (0-based).
        offset: Byte offset in the original data.
        size: Size in bytes of this slice.
    """

    index: int
    offset: int
    size: int


class XIOSplitter:
    """
    Splits large I/O operations into multiple smaller slices.

    This splitter divides a large MemoryObj into multiple slices that can be
    transferred in parallel using concurrent I/O operations.
    """

    def __init__(self, chunk_size: int):
        """
        Initialize the splitter.

        Args:
            chunk_size: The size in bytes to split each I/O into.
        """
        self.chunk_size = chunk_size

    def split(self, total_size: int) -> List[XIOSlice]:
        """
        Split a large I/O into multiple slices.

        Args:
            total_size: Total size in bytes to split.

        Returns:
            List of XIOSlice objects representing the split portions.
        """
        if total_size <= 0:
            return []

        slices: List[XIOSlice] = []
        offset = 0
        index = 0

        while offset < total_size:
            remaining = total_size - offset
            slice_size = min(self.chunk_size, remaining)

            slices.append(
                XIOSlice(
                    index=index,
                    offset=offset,
                    size=slice_size,
                )
            )

            offset += slice_size
            index += 1

        return slices

    def get_num_slices(self, total_size: int) -> int:
        """
        Calculate the number of slices for a given size.

        Args:
            total_size: Total size in bytes.

        Returns:
            Number of slices that would be created.
        """
        if total_size <= 0 or self.chunk_size <= 0:
            return 1 if total_size > 0 else 0

        return (total_size + self.chunk_size - 1) // self.chunk_size
