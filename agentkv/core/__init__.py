"""
agentkv.core — Core allocator internals.

This sub-package contains:
  - radix_tree.py      : CoW radix tree with fork semantics
  - slab_allocator.py  : Block pool with epoch-based ref counting
  - reclamation.py     : Safe memory reclamation policy
  - triton_kernels.py  : Triton device kernels (copy_block, zero_block)

Phase 1 implementation lives here.
"""

from agentkv.core.radix_tree import CoWRadixTree, NodeHandle
from agentkv.core.slab_allocator import SlabAllocator, BlockHandle

__all__ = [
    "CoWRadixTree",
    "NodeHandle",
    "SlabAllocator",
    "BlockHandle",
]
