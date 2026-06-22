"""
agentkv/core/reclamation.py — Epoch-based safe memory reclamation.

Why epoch-based reclamation?
------------------------------
The CoW radix tree creates a situation where a block can be *logically* freed
(its reference count drops to zero) while another thread/stream still holds a
pointer to it obtained during a prefix match.  This is the classic
"safe memory reclamation" problem in lock-free data structures.

We use a simplified, two-epoch scheme (inspired by Sievert et al.'s "Eras"
and the Linux kernel's RCU):

  Epoch 0 (current)  — blocks freed in this epoch are placed in pending_0.
  Epoch 1 (previous) — blocks moved from pending_0 when epoch advances.
  Retired queue      — blocks safe to return to the free list.

An epoch advance happens when:
  1. The allocator calls ``maybe_advance()`` (called every N alloc operations,
     controlled by PoolConfig.epoch_interval).
  2. All readers that started before the advance have declared quiescence
     (called ``leave_read_section()``).

In v1, "readers" are Python threads.  In v2, reader quiescence will be tracked
at the CUDA stream level using stream-ordered events.

Thread safety
-------------
All public methods acquire a fine-grained lock.  This is the correct, simple
baseline.  Contention is low because epoch advances are rare (every N allocs).

Reclamation strategy choice
---------------------------
We chose epoch-based over hazard pointers for two reasons:
  1. Hazard pointers require per-thread registration infrastructure that adds
     complexity for no benefit in our allocation-frequency range.
  2. Our "readers" (prefix-match callers) are short-lived (typically <1 ms),
     so epoch advances complete quickly.

Chosen NOT to use:
  - Reference counting alone: fine for simple trees but prone to cascading
    free storms when a shared root block's ref count hits zero and triggers
    a recursive teardown of thousands of descendant blocks.  We use ref counts
    as the *triggering* condition but defer the actual deallocation to the
    epoch mechanism.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, Set


class EpochReclaimer:
    """Two-epoch safe memory reclaimer.

    Parameters
    ----------
    free_callback : Callable[[int], None]
        Function called with a block_id when it is safe to return that block
        to the free pool.  Typically ``SlabAllocator._do_free``.
    """

    def __init__(self, free_callback: Callable[[int], None]) -> None:
        self._free_cb = free_callback
        self._lock = threading.Lock()

        # Current epoch counter (monotonically increasing)
        self._epoch: int = 0

        # Blocks pending retirement, keyed by the epoch in which they were freed.
        # deque of (epoch_freed, block_id) pairs.
        self._pending: Deque[tuple[int, int]] = deque()

        # Set of epoch values at which active readers started their read section.
        # A reader calls enter_read_section() → gets a token (the current epoch).
        # It calls leave_read_section(token) when done.
        self._active_reader_epochs: Set[int] = set()

        # Track total calls to maybe_advance for stats / testing.
        self._advance_count: int = 0
        self._freed_count: int = 0

    # ── Reader section API ────────────────────────────────────────────────────

    def enter_read_section(self) -> int:
        """Called by a prefix-match reader before accessing shared tree nodes.

        Returns an opaque token (the current epoch) that must be passed to
        ``leave_read_section`` when the read is complete.
        """
        with self._lock:
            token = self._epoch
            self._active_reader_epochs.add(token)
            return token

    def leave_read_section(self, token: int) -> None:
        """Called by a prefix-match reader after it has finished using tree nodes.

        Parameters
        ----------
        token : int
            The value returned by the matching ``enter_read_section()`` call.
        """
        with self._lock:
            self._active_reader_epochs.discard(token)
            # After a reader leaves, try to retire any pending blocks whose
            # epoch is now safe.
            self._try_retire()

    # ── Retire / advance API ─────────────────────────────────────────────────

    def retire(self, block_id: int) -> None:
        """Mark a block as logically freed.

        The block will be physically returned to the free pool once all readers
        that started before this call have declared quiescence.

        Parameters
        ----------
        block_id : int
            The block to retire.
        """
        with self._lock:
            self._pending.append((self._epoch, block_id))
            self._try_retire()

    def maybe_advance(self) -> None:
        """Advance the epoch counter if no readers hold the current epoch.

        Call this every ``PoolConfig.epoch_interval`` allocations.
        Advancing the epoch allows blocks retired in previous epochs to be
        reclaimed once readers from those epochs exit.
        """
        with self._lock:
            # Only advance if no reader is sitting in the current epoch.
            # Readers in older epochs are also safe to advance past, since they
            # will leave_read_section with their old token and trigger _try_retire.
            if self._epoch not in self._active_reader_epochs:
                self._epoch += 1
                self._advance_count += 1
            self._try_retire()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _try_retire(self) -> None:
        """Retire all pending blocks that are safe to free.

        A block retired in epoch E is safe when no active reader started in
        epoch ≤ E.  Since active_reader_epochs is a set of starting epochs,
        the safe threshold is: min(active_reader_epochs) > E, or the set is
        empty.

        Must be called with self._lock held.
        """
        if not self._pending:
            return

        if self._active_reader_epochs:
            safe_before_epoch = min(self._active_reader_epochs)
        else:
            # No active readers → all pending blocks are safe
            safe_before_epoch = self._epoch + 1

        while self._pending:
            freed_epoch, block_id = self._pending[0]
            if freed_epoch < safe_before_epoch:
                self._pending.popleft()
                self._free_cb(block_id)
                self._freed_count += 1
            else:
                break

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "epoch": self._epoch,
                "pending_blocks": len(self._pending),
                "active_readers": len(self._active_reader_epochs),
                "epoch_advances": self._advance_count,
                "blocks_freed": self._freed_count,
            }

    def pending_count(self) -> int:
        """Number of blocks waiting for safe reclamation."""
        with self._lock:
            return len(self._pending)
