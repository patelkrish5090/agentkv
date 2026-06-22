"""
agentkv — GPU-resident CoW radix-tree KV-cache memory manager.

Public API
----------
  AgentKVPool      : main user-facing pool class (Phase 2+)
  CoWRadixTree     : low-level radix tree (Phase 1, for advanced users)
  SlabAllocator    : low-level block pool (Phase 1, for advanced users)
  PoolConfig       : configuration dataclass
"""

__version__ = "0.1.0.dev0"
__author__ = "AgentKV Contributors"
__license__ = "Apache-2.0"

from agentkv.pool import AgentKVPool
from agentkv.core.radix_tree import CoWRadixTree, DualRadixTree
from agentkv.core.slab_allocator import SlabAllocator
from agentkv.core.config import PoolConfig

__all__ = [
    "__version__",
    "AgentKVPool",
    "CoWRadixTree",
    "DualRadixTree",
    "SlabAllocator",
    "PoolConfig",
]
