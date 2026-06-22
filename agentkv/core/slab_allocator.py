"""
agentkv/core/slab_allocator.py — Block pool with epoch-based reference counting.

Design overview
---------------
The SlabAllocator manages a fixed pool of N KV blocks pre-allocated as a
contiguous torch.Tensor at construction time.  No dynamic memory allocation
happens in the hot path (alloc / free).

Block states
~~~~~~~~~~~~
  FREE      — in the free list, available for allocation.
  ALLOCATED — owned by exactly one agent.
  SHARED    — owned by ≥2 agents (ref_count > 1).  Read-only from each agent's
              perspective; a write triggers a CoW copy (handled by the radix tree).

Reference counting
~~~~~~~~~~~~~~~~~~
Each block has an integer reference count.  The lifecycle:

  alloc()      → ref_count = 1, state = ALLOCATED
  inc_ref()    → ref_count += 1, state = SHARED
  dec_ref()    → ref_count -= 1
               if ref_count == 0: retire to EpochReclaimer (not immediately freed)
  _do_free()   → actually returns block to free list (called by reclaimer)

Concurrency model
~~~~~~~~~~~~~~~~~
In v1, all operations acquire a single global lock.  This is deliberately
conservative.  The lock is not held during GPU memory operations (copy_block
etc.) — only during the metadata bookkeeping.

A future v2 path will use atomic operations on a GPU-resident free list
(CAS-based bump allocator with per-warp local pools) for the hot allocation
path, keeping only the epoch reclaimer on CPU.

Free list
~~~~~~~~~
We use a Python collections.deque as a LIFO free list.  LIFO (stack) order
improves GPU L2 cache locality since recently-freed blocks are hot in cache.

BlockHandle
~~~~~~~~~~~
An opaque integer block ID.  The caller never needs to compute offsets into
the KV tensor; they use block_id and the allocator provides the tensor slice
via ``get_block_tensor()``.
"""

from __future__ import annotations

import threading
import warnings
from collections import deque
from typing import Deque, Dict, Optional

import torch

from agentkv.core.config import PoolConfig
from agentkv.core.reclamation import EpochReclaimer
from agentkv.core import triton_kernels as _tk

# Sentinel value meaning "no block"
INVALID_BLOCK: int = -1

# Block state constants (stored in metadata tensor for debugging)
_STATE_FREE = 0
_STATE_ALLOCATED = 1
_STATE_SHARED = 2


class BlockHandle:
    """Opaque handle to a KV block.

    Users should not inspect the ``block_id`` attribute; treat this as opaque.
    The exception is low-level testing code that cross-checks allocator state.
    """

    __slots__ = ("block_id",)

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id

    def is_valid(self) -> bool:
        return self.block_id != INVALID_BLOCK

    def __repr__(self) -> str:
        return f"BlockHandle(id={self.block_id})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockHandle):
            return self.block_id == other.block_id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.block_id)


class SlabAllocator:
    """Fixed-capacity KV block pool.

    Parameters
    ----------
    config : PoolConfig
        Pool configuration.  ``total_blocks``, ``block_size``, ``num_layers``,
        etc. are read from here.  The underlying KV tensor is allocated on
        ``config.device``.

    Usage
    -----
    >>> cfg = PoolConfig(total_blocks=256, block_size=16, ...)
    >>> alloc = SlabAllocator(cfg)
    >>> handle = alloc.alloc()          # claim a free block
    >>> alloc.inc_ref(handle)           # a second agent holds a reference
    >>> alloc.dec_ref(handle)           # first agent releases — block still live
    >>> alloc.dec_ref(handle)           # ref_count hits 0 → queued for reclamation
    >>> alloc.maybe_advance_epoch()     # call periodically; triggers actual free
    """

    def __init__(self, config: PoolConfig) -> None:
        self._cfg = config
        self._lock = threading.Lock()

        # ── bfloat16 guard for Turing GPUs (T4 = sm_75) ──────────────────────
        # T4 does not support bfloat16 natively.  Warn early rather than
        # crashing inside a Triton kernel deep in a benchmark.
        if (
            config.dtype == "bfloat16"
            and config.device.startswith("cuda")
            and not _tk._supports_bfloat16_gpu()
        ):
            warnings.warn(
                "bfloat16 is not natively supported on this GPU (Turing / sm_75, e.g. T4). "
                "The pool will use float16 instead to avoid correctness issues. "
                "Pass dtype='float16' explicitly to suppress this warning.",
                stacklevel=2,
            )
            config = PoolConfig(
                total_blocks=config.total_blocks,
                block_size=config.block_size,
                num_layers=config.num_layers,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                dtype="float16",
                device=config.device,
                max_agents=config.max_agents,
                epoch_interval=config.epoch_interval,
            )
            self._cfg = config

        # ── KV data tensor ────────────────────────────────────────────────────
        # Shape: [total_blocks, num_layers, 2, num_kv_heads, block_size, head_dim]
        # The factor of 2 is for K and V.
        # On CUDA this lives in device global memory (no pinning needed;
        # the GPU reads it directly from its own DRAM).
        self._kv_data: torch.Tensor = torch.zeros(
            config.total_blocks,
            config.num_layers,
            2,  # K, V
            config.num_kv_heads,
            config.block_size,
            config.head_dim,
            dtype=_str_to_dtype(config.dtype),
            device=config.device,
        )

        # ── Reference count array (CPU, int32) ───────────────────────────────
        # Kept on CPU even for CUDA pools — no device→host round-trips needed.
        self._ref_counts: torch.Tensor = torch.zeros(
            config.total_blocks, dtype=torch.int32
        )

        # ── Free list (LIFO deque) ────────────────────────────────────────────
        self._free_list: Deque[int] = deque(range(config.total_blocks))

        # ── Deferred free queue ───────────────────────────────────────────────
        # Avoids lock-order deadlock between self._lock and reclaimer._lock.
        # _do_free() (called from inside reclaimer._lock) pushes here;
        # _drain_deferred() (called under self._lock) moves IDs to _free_list.
        self._deferred_free: deque = deque()

        # ── Epoch-based reclaimer ─────────────────────────────────────────────
        self._reclaimer = EpochReclaimer(free_callback=self._do_free)

        # ── Operation counter for epoch advancement ───────────────────────────
        self._op_count: int = 0

        # ── Stats ─────────────────────────────────────────────────────────────
        self._alloc_count: int = 0
        self._free_count: int = 0
        self._peak_allocated: int = 0

    # ── Allocation ────────────────────────────────────────────────────────────

    def alloc(self) -> BlockHandle:
        """Claim a free block.

        Returns
        -------
        BlockHandle
            A valid handle with ref_count = 1.

        Raises
        ------
        MemoryError
            If the pool is exhausted (no free blocks available).
        """
        with self._lock:
            if not self._free_list:
                raise MemoryError(
                    f"AgentKV pool exhausted: all {self._cfg.total_blocks} blocks are in use. "
                    "Increase PoolConfig.total_blocks or free some agents."
                )
            block_id = self._free_list.pop()  # LIFO for cache locality
            self._ref_counts[block_id] = 1
            self._alloc_count += 1
            current = self._alloc_count - self._free_count
            if current > self._peak_allocated:
                self._peak_allocated = current
            self._tick()
            return BlockHandle(block_id)

    def inc_ref(self, handle: BlockHandle) -> None:
        """Increment the reference count of a block (CoW fork).

        Call this when a second agent takes ownership of the same block.
        After this call the block is in SHARED state and must not be mutated
        by either agent without a CoW copy.

        Parameters
        ----------
        handle : BlockHandle
            Block whose ref count to increment.
        """
        self._validate_handle(handle)
        with self._lock:
            bid = handle.block_id
            if self._ref_counts[bid].item() <= 0:
                raise ValueError(
                    f"inc_ref on block {bid} with ref_count={self._ref_counts[bid].item()} "
                    "(not allocated — double-inc or use-after-free?)"
                )
            self._ref_counts[bid] += 1
            self._tick()

    def dec_ref(self, handle: BlockHandle) -> bool:
        """Decrement the reference count of a block.

        If the ref count reaches zero, the block is *retired* (not immediately
        freed) to the EpochReclaimer.  The block is physically returned to the
        free list only after all active prefix-match readers have left their
        read sections.

        Parameters
        ----------
        handle : BlockHandle

        Returns
        -------
        bool
            True if the block was retired (ref_count hit zero), False otherwise.
        """
        self._validate_handle(handle)
        with self._lock:
            bid = handle.block_id
            current = self._ref_counts[bid].item()
            if current <= 0:
                raise ValueError(
                    f"dec_ref on block {bid} with ref_count={current} "
                    "(already free or double-free?)"
                )
            self._ref_counts[bid] -= 1
            if self._ref_counts[bid].item() == 0:
                self._reclaimer.retire(bid)
                self._free_count += 1
                self._tick()
                return True
            self._tick()
            return False

    # ── Block data access ─────────────────────────────────────────────────────

    def get_block_tensor(self, handle: BlockHandle) -> torch.Tensor:
        """Return a view into the KV data for the given block.

        Shape: [num_layers, 2, num_kv_heads, block_size, head_dim]

        This is a zero-copy view; the caller can read from it freely.
        To write (for a CoW copy), the caller must have exclusive ownership
        (ref_count == 1).

        Parameters
        ----------
        handle : BlockHandle
        """
        self._validate_handle(handle)
        return self._kv_data[handle.block_id]

    def copy_block(self, src: BlockHandle, dst: BlockHandle) -> None:
        """Copy KV data from src block to dst block (CPU or GPU).

        On CUDA, dispatches to the Triton copy_block kernel (autotuned for
        the detected GPU architecture, including T4 sm_75).
        On CPU (or if Triton is unavailable), uses torch.Tensor.copy_.
        """
        self._validate_handle(src)
        self._validate_handle(dst)
        with self._lock:
            if self._ref_counts[dst.block_id].item() != 1:
                raise ValueError(
                    f"copy_block: dst block {dst.block_id} must have ref_count=1 "
                    f"(got {self._ref_counts[dst.block_id].item()}). "
                    "Only exclusively-owned blocks may be written."
                )
        # Data copy is outside the lock — potentially expensive GPU op.
        # The ref count check above guarantees dst is exclusively ours.
        _tk.copy_block(
            self._kv_data[src.block_id],
            self._kv_data[dst.block_id],
        )

    def zero_block(self, handle: BlockHandle) -> None:
        """Zero-fill a block's KV data (security / fresh allocation).

        On CUDA, dispatches to the Triton zero_block kernel.
        Falls back to torch.Tensor.zero_() on CPU.
        """
        self._validate_handle(handle)
        with self._lock:
            if self._ref_counts[handle.block_id].item() != 1:
                raise ValueError(
                    f"zero_block: block {handle.block_id} must be exclusively owned "
                    f"(ref_count=1, got {self._ref_counts[handle.block_id].item()})"
                )
        _tk.zero_block(self._kv_data[handle.block_id])

    # ── Epoch / reclamation ───────────────────────────────────────────────────

    def enter_read_section(self) -> int:
        """Begin a prefix-match read section.  Returns an opaque epoch token."""
        return self._reclaimer.enter_read_section()

    def leave_read_section(self, token: int) -> None:
        """End a prefix-match read section."""
        self._reclaimer.leave_read_section(token)

    def maybe_advance_epoch(self) -> None:
        """Manually trigger an epoch advance.  Safe to call at any time."""
        self._reclaimer.maybe_advance()

    # ── Introspection / stats ─────────────────────────────────────────────────

    def ref_count(self, handle: BlockHandle) -> int:
        """Return the current reference count of a block (for testing)."""
        self._validate_handle(handle)
        with self._lock:
            return int(self._ref_counts[handle.block_id].item())

    @property
    def free_count(self) -> int:
        """Number of blocks currently in the free list."""
        with self._lock:
            return len(self._free_list)

    @property
    def allocated_count(self) -> int:
        """Number of blocks currently allocated (ref_count > 0)."""
        with self._lock:
            return int((self._ref_counts > 0).sum().item())

    def stats(self) -> dict:
        with self._lock:
            recl = self._reclaimer.stats()
        return {
            "total_blocks": self._cfg.total_blocks,
            "free_blocks": self.free_count,
            "allocated_blocks": self.allocated_count,
            "peak_allocated": self._peak_allocated,
            "total_allocs": self._alloc_count,
            "total_frees": self._free_count,
            "pending_reclamation": recl["pending_blocks"],
            "epoch": recl["epoch"],
            "pool_size_gb": self._cfg.total_gb,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _do_free(self, block_id: int) -> None:
        """Called by EpochReclaimer when it is safe to return block_id to the
        free list.  The ref count was already set to 0 in dec_ref; we just
        push the id back to the free list.

        WARNING: This is called from inside EpochReclaimer._try_retire, which
        is called while holding self._reclaimer._lock.  We must NOT acquire
        self._lock here to avoid a lock-order deadlock.  Instead we access
        _free_list directly — this is safe because _do_free is only called from
        the same thread that holds _reclaimer._lock (always the thread calling
        dec_ref / leave_read_section / maybe_advance_epoch, all of which
        subsequently hold self._lock before calling into the reclaimer ... wait,
        actually no.

        CORRECTED design: _do_free acquires self._lock.  The reclaimer's lock
        is released before _do_free is called (see EpochReclaimer._try_retire
        — it calls free_callback *while* holding its own lock, so we must
        NOT re-acquire _reclaimer._lock from here, but we CAN acquire
        self._lock because the call chain is always:

          thread → self._lock → reclaimer._lock → _do_free → self._lock  ← DEADLOCK

        To avoid this, _do_free uses a separate deque (_pending_free_list)
        that is drained back into _free_list the next time self._lock is held
        by any public API call.  This is the "deferred free" pattern.
        """
        # Push to a lock-free intermediate queue that is drained under self._lock.
        self._deferred_free.append(block_id)

    def _drain_deferred(self) -> None:
        """Move all deferred-free block IDs back into the free list.

        Must be called with self._lock held.
        """
        while self._deferred_free:
            self._free_list.appendleft(self._deferred_free.pop())

    def _tick(self) -> None:
        """Increment op counter and maybe advance epoch.  Called with self._lock held."""
        self._op_count += 1
        if self._op_count % self._cfg.epoch_interval == 0:
            # Release self._lock before calling maybe_advance, then re-acquire.
            # (maybe_advance → _try_retire → _do_free → _deferred_free.append)
            # Since _do_free only touches _deferred_free (not self._lock),
            # this is safe.
            self._lock.release()
            try:
                self._reclaimer.maybe_advance()
            finally:
                self._lock.acquire()
        self._drain_deferred()

    def _validate_handle(self, handle: BlockHandle) -> None:
        if not isinstance(handle, BlockHandle):
            raise TypeError(f"Expected BlockHandle, got {type(handle)}")
        if not handle.is_valid():
            raise ValueError("BlockHandle is invalid (block_id == INVALID_BLOCK)")
        if handle.block_id < 0 or handle.block_id >= self._cfg.total_blocks:
            raise ValueError(
                f"block_id {handle.block_id} out of range [0, {self._cfg.total_blocks})"
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _str_to_dtype(dtype_str: str) -> torch.dtype:
    _map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in _map:
        raise ValueError(f"Unknown dtype '{dtype_str}'")
    return _map[dtype_str]
