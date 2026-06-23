"""
agentkv/core/radix_tree.py — Copy-on-Write Radix Tree with fork semantics.

Architecture: DualRadixTree
---------------------------
The DualRadixTree maintains two logical layers:

  1. Shared (base) tree  — read-only prefix cache shared across all agents.
     An agent promotes its tokens into this layer by calling ``commit_prefix()``.
     Once in the shared tree, a block is ref-counted; any agent can read it.

  2. Per-agent residual  — each agent (identified by a NodeHandle) has its own
     sequence of block IDs for tokens *beyond* the shared prefix.

This design is the core memory efficiency win: N agents forked from the same
parent share the parent's blocks via the shared tree, consuming O(shared_prefix)
GPU memory instead of O(N × shared_prefix).

CoW semantics
~~~~~~~~~~~~~
``fork(parent_handle)`` creates a child that shares all of the parent's current
blocks (both shared-tree blocks and residual blocks).  For shared-tree blocks,
a ref-count increment is sufficient.  For residual blocks, we inc_ref on each
block the child inherits.  When either parent or child appends new tokens, they
allocate fresh blocks (no conflict).  When either tries to *modify* a block it
shares (which in LLM inference only happens via recomputation / eviction, not
normal forward passes), it must CoW-copy first.

Node metadata
~~~~~~~~~~~~~
Each node (agent handle) stores:
  - shared_node_id   : ID into the shared radix tree (None if no shared prefix)
  - shared_match_len : number of tokens covered by the shared prefix
  - residual_blocks  : list of BlockHandle for tokens beyond the shared prefix
  - tokens           : the full token sequence this agent has seen
  - ref_count        : how many "live" references to this handle exist
                       (separate from block-level ref counts)

Token storage
~~~~~~~~~~~~~
We store the token sequence (list of ints) on the CPU side.  Token sequences
are tiny compared to KV blocks (a 2048-token sequence is ~4 KB of int32 vs.
potentially gigabytes of KV data).

Radix tree node matching
~~~~~~~~~~~~~~~~~~~~~~~~~
Prefix matching uses the standard radix tree longest-common-prefix algorithm.
Children are stored in a dict keyed by the first token of their edge label.
We use a flat array of SharedNode objects (indexed by node_id) for O(1) lookup.

Thread safety
~~~~~~~~~~~~~
All tree mutations (insert, fork, free) acquire a global tree lock.
Prefix match (read path) uses the EpochReclaimer reader section to protect
against concurrent frees.

v1 Limitation
~~~~~~~~~~~~~
The radix tree metadata is CPU-resident.  GPU kernels (attention) receive a
list of block IDs and access the KV tensor via the SlabAllocator.
A future v2 GPU-resident tree will be pointer-linked directly in CUDA global
memory with atomic-CAS operations on the child pointers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agentkv.core.slab_allocator import BlockHandle, SlabAllocator
from agentkv.core.config import PoolConfig

# Sentinel node id
ROOT_NODE_ID: int = 0
INVALID_NODE_ID: int = -1


# ── Shared tree nodes ─────────────────────────────────────────────────────────

@dataclass
class SharedNode:
    """A node in the shared (read-only) radix tree.

    Attributes
    ----------
    node_id     : unique integer ID (index into DualRadixTree._shared_nodes).
    parent_id   : parent node ID, or INVALID_NODE_ID for the root.
    edge_tokens : the token subsequence on the edge from parent → this node.
    blocks      : ordered list of BlockHandles covering edge_tokens.
                  len(blocks) == ceil(len(edge_tokens) / block_size).
    children    : dict mapping first-token-of-edge → child node_id.
    ref_count   : number of active agents that hold this node as their
                  shared_node_id (or that have a child referencing it through
                  the common-prefix chain).
    """

    node_id: int
    parent_id: int
    edge_tokens: List[int] = field(default_factory=list)
    blocks: List[BlockHandle] = field(default_factory=list)
    children: Dict[int, int] = field(default_factory=dict)  # first_token → child_id
    ref_count: int = 0


# ── Per-agent handle ──────────────────────────────────────────────────────────

@dataclass
class NodeHandle:
    """Per-agent state in the DualRadixTree.

    This is the opaque handle returned to callers (and to the AgentKVPool).
    Callers should not inspect its fields; use the DualRadixTree API instead.
    """

    handle_id: int
    shared_node_id: int  # INVALID_NODE_ID if no shared prefix matched
    shared_match_len: int  # tokens covered by shared prefix
    residual_blocks: List[BlockHandle] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    ref_count: int = 1  # live references to this handle (not the blocks)
    is_freed: bool = False

    def __repr__(self) -> str:
        return (
            f"NodeHandle("
            f"id={self.handle_id}, "
            f"shared_node={self.shared_node_id}, "
            f"shared_len={self.shared_match_len}, "
            f"residual_blocks={len(self.residual_blocks)}, "
            f"tokens={len(self.tokens)}"
            f")"
        )


# ── DualRadixTree ─────────────────────────────────────────────────────────────

class DualRadixTree:
    """Copy-on-Write radix tree KV cache manager.

    Parameters
    ----------
    allocator : SlabAllocator
        The underlying block pool.
    config : PoolConfig
        Pool configuration (block_size used for edge partitioning).

    Public API
    ----------
    create_root(tokens)              → NodeHandle
    fork(parent_handle)              → NodeHandle (child shares parent's blocks)
    allocate_block(handle)           → BlockHandle (fresh block for new tokens)
    append_tokens(handle, tokens)    → None (record new tokens; may allocate)
    free(handle)                     → None
    match_prefix(tokens)             → (NodeHandle | None, match_length)
    commit_prefix(handle, length)    → None (promote tokens to shared tree)
    get_block_ids(handle)            → List[int] (for GPU attention)
    stats()                          → dict
    """

    def __init__(self, allocator: SlabAllocator, config: PoolConfig) -> None:
        self._alloc = allocator
        self._cfg = config
        self._lock = threading.Lock()

        # Shared radix tree nodes, indexed by node_id.
        # Root node (id=ROOT_NODE_ID=0) is pre-allocated and always exists.
        self._shared_nodes: Dict[int, SharedNode] = {
            ROOT_NODE_ID: SharedNode(
                node_id=ROOT_NODE_ID,
                parent_id=INVALID_NODE_ID,
                edge_tokens=[],
                blocks=[],
                ref_count=0,
            )
        }
        self._next_node_id: int = 1  # monotonically increasing

        # Per-agent handles, indexed by handle_id.
        self._handles: Dict[int, NodeHandle] = {}
        self._next_handle_id: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def create_root(self, tokens: List[int]) -> NodeHandle:
        """Create a root agent handle from a list of prompt tokens.

        No blocks are allocated here — blocks are allocated lazily via
        ``allocate_block`` or as part of ``append_tokens``.  This keeps the
        hot path allocation-free for pure prefix-match scenarios.

        Parameters
        ----------
        tokens : List[int]
            The full prompt token sequence for this agent.

        Returns
        -------
        NodeHandle
        """
        with self._lock:
            # Try to match against the shared tree first.
            shared_node_id, match_len = self._match_prefix_locked(tokens)

            handle = NodeHandle(
                handle_id=self._next_handle_id,
                shared_node_id=shared_node_id,
                shared_match_len=match_len,
                residual_blocks=[],
                tokens=list(tokens),
                ref_count=1,
            )
            self._next_handle_id += 1

            # Increment ref count on the matched shared node.
            if shared_node_id != INVALID_NODE_ID:
                self._inc_shared_ref_locked(shared_node_id)

            self._handles[handle.handle_id] = handle
            return handle

    def fork(self, parent: NodeHandle) -> NodeHandle:
        """Fork a child agent from a parent.

        The child shares ALL of the parent's current blocks:
          - Shared-tree blocks: inc_ref on the shared node (no block-level copy).
          - Residual blocks: inc_ref on each individual BlockHandle.

        After forking, both parent and child can independently append new tokens
        (each gets its own fresh blocks).  Neither can modify shared blocks
        without a CoW copy (enforced by SlabAllocator.copy_block).

        Parameters
        ----------
        parent : NodeHandle
            The parent handle.  Must not be freed.

        Returns
        -------
        NodeHandle
            A new child handle.
        """
        with self._lock:
            self._validate_handle_locked(parent)

            # Inc ref on residual blocks (block-level sharing)
            for bh in parent.residual_blocks:
                self._alloc.inc_ref(bh)

            # Inc ref on shared tree node
            if parent.shared_node_id != INVALID_NODE_ID:
                self._inc_shared_ref_locked(parent.shared_node_id)

            child = NodeHandle(
                handle_id=self._next_handle_id,
                shared_node_id=parent.shared_node_id,
                shared_match_len=parent.shared_match_len,
                residual_blocks=list(parent.residual_blocks),  # shallow copy of list
                tokens=list(parent.tokens),
                ref_count=1,
            )
            self._next_handle_id += 1
            self._handles[child.handle_id] = child
            return child

    def allocate_block(self, handle: NodeHandle) -> BlockHandle:
        """Allocate a fresh block for exclusive use by the agent.

        This block is appended to the agent's residual block list.
        It has ref_count=1 and is safe to write.

        Parameters
        ----------
        handle : NodeHandle

        Returns
        -------
        BlockHandle
        """
        with self._lock:
            self._validate_handle_locked(handle)
            bh = self._alloc.alloc()
            handle.residual_blocks.append(bh)
            return bh

    def ensure_mutable_block(self, handle: NodeHandle, block_idx: int) -> BlockHandle:
        """Ensure that the residual block at `block_idx` is exclusively owned.
        
        If the block is shared (ref_count > 1), a CoW copy is performed:
        a new block is allocated, the data is copied, and the new block replaces
        the shared one in the agent's residual list.
        
        Parameters
        ----------
        handle : NodeHandle
        block_idx : int
            Index into handle.residual_blocks
            
        Returns
        -------
        BlockHandle
            The (possibly new) exclusively owned block.
        """
        with self._lock:
            self._validate_handle_locked(handle)
            bh = handle.residual_blocks[block_idx]
            if self._alloc.ref_count(bh) > 1:
                # Need CoW
                new_bh = self._alloc.alloc()
                self._alloc.copy_block(bh, new_bh)
                self._alloc.dec_ref(bh)
                handle.residual_blocks[block_idx] = new_bh
                return new_bh
            return bh

    def free_block(self, handle: NodeHandle, block: BlockHandle) -> None:
        """Free a specific residual block held by an agent.

        If the block is shared (ref_count > 1), only decrements the ref count.
        Physical reclamation is deferred to the EpochReclaimer.

        Parameters
        ----------
        handle : NodeHandle
        block : BlockHandle
        """
        with self._lock:
            self._validate_handle_locked(handle)
            if block not in handle.residual_blocks:
                raise ValueError(
                    f"block {block} is not a residual block of handle {handle.handle_id}"
                )
            handle.residual_blocks.remove(block)
            self._alloc.dec_ref(block)

    def append_tokens(self, handle: NodeHandle, new_tokens: List[int]) -> None:
        """Record new tokens for an agent.

        This does NOT allocate blocks — call ``allocate_block`` explicitly
        for each new block needed.  This method only updates the token list.

        Parameters
        ----------
        handle : NodeHandle
        new_tokens : List[int]
        """
        with self._lock:
            self._validate_handle_locked(handle)
            handle.tokens.extend(new_tokens)

    def free(self, handle: NodeHandle) -> None:
        """Free all resources associated with an agent handle.

        - Decrements ref counts on all residual blocks (→ reclaimer if → 0).
        - Decrements ref count on the shared tree node (may collapse empty nodes).
        - Removes the handle from the registry.

        Parameters
        ----------
        handle : NodeHandle
        """
        with self._lock:
            self._validate_handle_locked(handle)
            if handle.is_freed:
                raise ValueError(f"double-free of handle {handle.handle_id}")

            # Free residual blocks
            for bh in handle.residual_blocks:
                self._alloc.dec_ref(bh)
            handle.residual_blocks.clear()

            # Dec ref on shared node
            if handle.shared_node_id != INVALID_NODE_ID:
                self._dec_shared_ref_locked(handle.shared_node_id)

            handle.is_freed = True
            del self._handles[handle.handle_id]

    def match_prefix(
        self, tokens: List[int]
    ) -> Tuple[Optional[NodeHandle], int]:
        """Find the longest shared-tree prefix matching ``tokens``.

        Uses the EpochReclaimer read section to protect against concurrent
        frees of shared nodes.

        Parameters
        ----------
        tokens : List[int]

        Returns
        -------
        (handle_or_None, match_length)
            If a match is found, creates a new NodeHandle pointing at the
            matched shared node.  Returns (None, 0) if no prefix matched.
        """
        epoch_token = self._alloc.enter_read_section()
        try:
            with self._lock:
                node_id, match_len = self._match_prefix_locked(tokens)
                if node_id == INVALID_NODE_ID or match_len == 0:
                    return None, 0

                # Create a temporary handle referencing the matched node.
                self._inc_shared_ref_locked(node_id)
                h = NodeHandle(
                    handle_id=self._next_handle_id,
                    shared_node_id=node_id,
                    shared_match_len=match_len,
                    tokens=list(tokens[:match_len]),
                )
                self._next_handle_id += 1
                self._handles[h.handle_id] = h
                return h, match_len
        finally:
            self._alloc.leave_read_section(epoch_token)

    def commit_prefix(self, handle: NodeHandle, length: int) -> None:
        """Promote the first ``length`` tokens of an agent's sequence into the
        shared radix tree, making them available for future prefix matches.

        This is the write path into the shared tree.  After commit, the promoted
        blocks become shared (ref_count incremented for the new shared node).
        The agent's residual list is updated to remove the promoted blocks.

        Parameters
        ----------
        handle : NodeHandle
        length : int
            Number of tokens to promote (must be a multiple of block_size,
            and ≤ len(residual_blocks) * block_size).
        """
        with self._lock:
            self._validate_handle_locked(handle)
            bs = self._cfg.block_size
            if length % bs != 0:
                raise ValueError(
                    f"commit_prefix: length={length} must be a multiple of "
                    f"block_size={bs}"
                )
            n_blocks = length // bs
            if n_blocks > len(handle.residual_blocks):
                raise ValueError(
                    f"commit_prefix: requested {n_blocks} blocks but agent only "
                    f"has {len(handle.residual_blocks)} residual blocks"
                )

            # Tokens to promote
            residual_tokens = handle.tokens[handle.shared_match_len:]
            promote_tokens = residual_tokens[:length]
            promote_blocks = handle.residual_blocks[:n_blocks]

            # Insert into shared tree
            new_node_id = self._insert_shared_locked(
                parent_id=handle.shared_node_id
                if handle.shared_node_id != INVALID_NODE_ID
                else ROOT_NODE_ID,
                tokens=promote_tokens,
                blocks=promote_blocks,
            )

            # Update handle: residual shrinks, shared grows
            handle.residual_blocks = handle.residual_blocks[n_blocks:]
            handle.shared_node_id = new_node_id
            handle.shared_match_len += length

            # Inc ref on new shared node for this agent
            self._inc_shared_ref_locked(new_node_id)

    def get_block_ids(self, handle: NodeHandle) -> List[int]:
        """Return the ordered list of all block IDs for this agent.

        This is what the attention kernel needs: a flat sequence of block IDs
        covering [shared_prefix_blocks..., residual_blocks...].

        Parameters
        ----------
        handle : NodeHandle

        Returns
        -------
        List[int]
        """
        with self._lock:
            self._validate_handle_locked(handle)
            ids: List[int] = []

            # Collect shared tree blocks (walk from shared node to root)
            if handle.shared_node_id != INVALID_NODE_ID:
                chain = self._collect_shared_chain_locked(handle.shared_node_id)
                for node in chain:
                    ids.extend(bh.block_id for bh in node.blocks)

            # Add residual blocks
            ids.extend(bh.block_id for bh in handle.residual_blocks)
            return ids

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            total_shared_blocks = sum(
                len(n.blocks) for n in self._shared_nodes.values()
            )
            return {
                "active_handles": len(self._handles),
                "shared_nodes": len(self._shared_nodes),
                "shared_blocks": total_shared_blocks,
            }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _match_prefix_locked(
        self, tokens: List[int]
    ) -> Tuple[int, int]:
        """Find the longest matching shared-tree prefix.

        Returns (node_id, match_length).  match_length is in tokens.
        Must be called with self._lock held.
        """
        current_id = ROOT_NODE_ID
        matched = 0

        while matched < len(tokens):
            node = self._shared_nodes[current_id]
            next_token = tokens[matched]

            if next_token not in node.children:
                break

            child_id = node.children[next_token]
            child = self._shared_nodes[child_id]
            edge = child.edge_tokens

            # How many tokens of the edge match?
            edge_match = 0
            for i, et in enumerate(edge):
                if matched + i >= len(tokens):
                    break
                if tokens[matched + i] != et:
                    break
                edge_match += 1

            if edge_match == 0:
                break

            matched += edge_match
            current_id = child_id

            if edge_match < len(edge):
                # Partial edge match — stop here
                break

        # Round down to block boundary
        bs = self._cfg.block_size
        matched = (matched // bs) * bs

        if matched == 0:
            return INVALID_NODE_ID, 0
        return current_id, matched

    def _insert_shared_locked(
        self, parent_id: int, tokens: List[int], blocks: List[BlockHandle]
    ) -> int:
        """Insert tokens+blocks as a new child of parent in the shared tree.

        Returns the new node_id.  Must be called with self._lock held.
        """
        parent = self._shared_nodes[parent_id]
        first_token = tokens[0] if tokens else None

        if first_token is None:
            return parent_id

        # Check if a child with this first token already exists (split needed)
        if first_token in parent.children:
            # Existing child — would need edge splitting for a full implementation.
            # For v1: overwrite only if the tokens exactly match (no split).
            # TODO: implement edge splitting for full radix tree correctness.
            child_id = parent.children[first_token]
            child = self._shared_nodes[child_id]
            if child.edge_tokens == tokens:
                # Update blocks in place (if they changed)
                child.blocks = list(blocks)
                return child_id
            # Fall through: create a new sibling (simplified v1 behavior)

        new_id = self._next_node_id
        self._next_node_id += 1
        new_node = SharedNode(
            node_id=new_id,
            parent_id=parent_id,
            edge_tokens=list(tokens),
            blocks=list(blocks),
            ref_count=0,
        )
        self._shared_nodes[new_id] = new_node
        parent.children[tokens[0]] = new_id
        return new_id

    def _inc_shared_ref_locked(self, node_id: int) -> None:
        """Increment ref count on a shared node and all its ancestors."""
        nid = node_id
        while nid != INVALID_NODE_ID and nid in self._shared_nodes:
            self._shared_nodes[nid].ref_count += 1
            nid = self._shared_nodes[nid].parent_id

    def _dec_shared_ref_locked(self, node_id: int) -> None:
        """Decrement ref count on a shared node and prune if ref_count → 0."""
        nid = node_id
        while nid != INVALID_NODE_ID and nid in self._shared_nodes:
            node = self._shared_nodes[nid]
            node.ref_count -= 1
            if node.ref_count <= 0 and nid != ROOT_NODE_ID and not node.children:
                # Prune leaf node — free its blocks
                for bh in node.blocks:
                    self._alloc.dec_ref(bh)
                parent_id = node.parent_id
                if parent_id in self._shared_nodes:
                    parent = self._shared_nodes[parent_id]
                    # Remove this child from parent
                    parent.children = {
                        k: v
                        for k, v in parent.children.items()
                        if v != nid
                    }
                del self._shared_nodes[nid]
                nid = parent_id
            else:
                nid = self._shared_nodes[nid].parent_id

    def _collect_shared_chain_locked(self, node_id: int) -> List[SharedNode]:
        """Return the chain of shared nodes from root to node_id (inclusive),
        in root-first order.  Used by get_block_ids.
        """
        chain: List[SharedNode] = []
        nid = node_id
        while nid != INVALID_NODE_ID and nid in self._shared_nodes:
            node = self._shared_nodes[nid]
            chain.append(node)
            nid = node.parent_id
            if nid == ROOT_NODE_ID:
                break  # Root has no blocks; stop here
        chain.reverse()
        return chain

    def _validate_handle_locked(self, handle: NodeHandle) -> None:
        if handle.handle_id not in self._handles:
            raise ValueError(
                f"NodeHandle {handle.handle_id} is not registered "
                "(was it already freed?)"
            )
        if handle.is_freed:
            raise ValueError(f"NodeHandle {handle.handle_id} has already been freed")


# ── CoWRadixTree: public alias for the canonical implementation ───────────────
# DualRadixTree IS the CoW radix tree.  This alias is for ergonomic imports.
CoWRadixTree = DualRadixTree
