"""
agentkv/pool.py — AgentKVPool: the main user-facing API (Phase 2).

This is the "developer-facing surface" described in the spec:
  "a developer can pip install agentkv and run a 10-line script that forks
   caches and reports memory usage, with no GPU code visible to them."

Design
------
AgentKVPool is a thin facade over DualRadixTree + SlabAllocator.  It:
  1. Constructs the PoolConfig from high-level parameters (capacity_gb, block_size).
  2. Instantiates the SlabAllocator (pre-allocates GPU memory).
  3. Instantiates the DualRadixTree.
  4. Exposes a clean, Pythonic API (no internal types leak out).

All GPU complexity is hidden behind this interface.  Callers work with
opaque handle objects and never see BlockHandle, SlabAllocator, etc.
"""

from __future__ import annotations

from typing import List, Optional

from agentkv.core.config import PoolConfig
from agentkv.core.slab_allocator import SlabAllocator
from agentkv.core.radix_tree import DualRadixTree, NodeHandle


class AgentKVPool:
    """GPU-resident CoW KV-cache memory pool for agentic LLM workloads.

    Quickstart
    ----------
    >>> pool = AgentKVPool(capacity_gb=40, block_size=16)
    >>> root = pool.create_root(prompt_tokens=[1, 2, 3, ...])
    >>> child_a = pool.fork(root)
    >>> child_b = pool.fork(root)
    >>> pool.allocate_block(child_a)
    >>> print(pool.stats())
    >>> pool.free(child_a)
    >>> pool.free(child_b)
    >>> pool.free(root)

    Parameters
    ----------
    capacity_gb : float
        Target GPU memory budget in GiB.  The actual pool size will be the
        largest power-of-two number of blocks that fits within this budget.
        Use ``PoolConfig.from_capacity_gb`` for the exact calculation.
    block_size : int
        Tokens per KV block (default 16).  Must match the serving framework's
        block size if using a framework integration.
    num_layers : int
        Number of transformer layers.
    num_kv_heads : int
        Number of KV attention heads per layer.
    head_dim : int
        Dimension per head.
    dtype : str
        KV cache dtype: 'float16', 'bfloat16', or 'float32'.
    device : str
        PyTorch device string (e.g. 'cuda', 'cuda:0', 'cpu').
    config : PoolConfig, optional
        If provided, overrides all other parameters.  Use this to pass a
        fully-configured PoolConfig directly.
    """

    def __init__(
        self,
        capacity_gb: float = 40.0,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: str = "float16",
        device: str = "cuda",
        *,
        config: Optional[PoolConfig] = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            self._cfg = PoolConfig.from_capacity_gb(
                capacity_gb=capacity_gb,
                block_size=block_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )

        self._allocator = SlabAllocator(self._cfg)
        self._tree = DualRadixTree(self._allocator, self._cfg)

    # ── Core API ──────────────────────────────────────────────────────────────

    def create_root(self, prompt_tokens: List[int]) -> NodeHandle:
        """Create a new root agent handle.

        The agent will share any common prefix already in the pool's shared
        radix tree (from a previous ``commit_prefix`` call).

        Parameters
        ----------
        prompt_tokens : List[int]
            The full prompt token sequence (vocabulary IDs).

        Returns
        -------
        NodeHandle
            An opaque handle.  Pass this to ``fork``, ``allocate_block``,
            ``free``, etc.
        """
        return self._tree.create_root(prompt_tokens)

    def fork(self, parent: NodeHandle) -> NodeHandle:
        """Fork a child agent from a parent.

        The child inherits all of the parent's KV blocks via ref-count sharing.
        No GPU memory is copied.  The child gets its own "residual" space for
        new tokens.

        Parameters
        ----------
        parent : NodeHandle

        Returns
        -------
        NodeHandle
            The child handle.  Independent from the parent: new blocks
            allocated by the child don't appear in the parent's view.
        """
        return self._tree.fork(parent)

    def allocate_block(self, handle: NodeHandle) -> None:
        """Allocate a fresh KV block for the agent's next tokens.

        The block is appended to the agent's residual block list.
        Call this once per ``block_size`` generated tokens.

        Parameters
        ----------
        handle : NodeHandle
        """
        self._tree.allocate_block(handle)

    def append_tokens(self, handle: NodeHandle, tokens: List[int]) -> None:
        """Record new generated tokens for an agent.

        This is a metadata-only operation (no block allocation).  Call
        ``allocate_block`` when crossing a block boundary.

        Parameters
        ----------
        handle : NodeHandle
        tokens : List[int]
        """
        self._tree.append_tokens(handle, tokens)

    def commit_prefix(self, handle: NodeHandle, length: int) -> None:
        """Promote the first ``length`` tokens to the shared radix tree.

        After this call, future agents with the same prefix will reuse the
        promoted KV blocks without copying.

        Parameters
        ----------
        handle : NodeHandle
        length : int
            Number of tokens to promote.  Must be a multiple of block_size.
        """
        self._tree.commit_prefix(handle, length)

    def free(self, handle: NodeHandle) -> None:
        """Release all resources held by an agent.

        Decrements ref counts on all owned blocks.  Blocks are reclaimed
        asynchronously by the epoch reclaimer once all readers exit.

        Parameters
        ----------
        handle : NodeHandle
        """
        self._tree.free(handle)

    def get_block_ids(self, handle: NodeHandle) -> List[int]:
        """Return the ordered list of block IDs for this agent.

        This is the tensor index sequence needed by the attention kernel:
        [shared_prefix_block_ids..., residual_block_ids...]

        Parameters
        ----------
        handle : NodeHandle

        Returns
        -------
        List[int]
        """
        return self._tree.get_block_ids(handle)

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a dict of pool-wide statistics.

        Includes:
          - total_blocks, free_blocks, allocated_blocks
          - peak_allocated (high-water mark)
          - active_handles (number of live agent handles)
          - shared_nodes, shared_blocks (in the shared radix tree)
          - pool_size_gb
          - pending_reclamation (blocks waiting for safe reclamation)
        """
        alloc_stats = self._allocator.stats()
        tree_stats = self._tree.stats()
        return {**alloc_stats, **tree_stats}

    @property
    def config(self) -> PoolConfig:
        """The pool configuration (read-only)."""
        return self._cfg

    @property
    def block_size(self) -> int:
        """Tokens per KV block."""
        return self._cfg.block_size

    @property
    def free_blocks(self) -> int:
        """Number of available (unallocated) blocks."""
        return self._allocator.free_count

    @property
    def allocated_blocks(self) -> int:
        """Number of currently allocated blocks."""
        return self._allocator.allocated_count

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"AgentKVPool("
            f"total={s['total_blocks']}, "
            f"free={s['free_blocks']}, "
            f"allocated={s['allocated_blocks']}, "
            f"agents={s['active_handles']}, "
            f"pool={self._cfg.total_gb:.2f} GiB"
            f")"
        )
