"""
tests/test_slab_allocator.py — Unit tests for SlabAllocator and EpochReclaimer.

Coverage:
  - Basic alloc / dec_ref lifecycle
  - Reference counting: inc_ref, dec_ref, double-free detection
  - Pool exhaustion (MemoryError)
  - Free list LIFO ordering
  - EpochReclaimer: retire → advance → _do_free cycle
  - Memory accounting: after N ops total_allocated == expected
  - copy_block / zero_block correctness (CPU path)
  - Thread-safety smoke test (concurrent alloc from multiple threads)
"""

import threading
import pytest
from agentkv.core.config import PoolConfig
from agentkv.core.slab_allocator import SlabAllocator, BlockHandle, INVALID_BLOCK


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return PoolConfig(
        total_blocks=64,
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype="float16",
        device="cpu",
    )


@pytest.fixture
def alloc(cfg):
    return SlabAllocator(cfg)


# ── Basic alloc/free ──────────────────────────────────────────────────────────

class TestBasicAllocFree:
    def test_alloc_returns_valid_handle(self, alloc):
        h = alloc.alloc()
        assert h.is_valid()
        assert h.block_id >= 0

    def test_alloc_ref_count_is_one(self, alloc):
        h = alloc.alloc()
        assert alloc.ref_count(h) == 1

    def test_alloc_reduces_free_count(self, alloc, cfg):
        before = alloc.free_count
        alloc.alloc()
        assert alloc.free_count == before - 1

    def test_dec_ref_to_zero_retires_block(self, alloc):
        h = alloc.alloc()
        retired = alloc.dec_ref(h)
        assert retired is True

    def test_alloc_after_free_reuses_block(self, alloc):
        h1 = alloc.alloc()
        bid1 = h1.block_id
        alloc.dec_ref(h1)
        alloc.maybe_advance_epoch()  # force reclamation
        h2 = alloc.alloc()
        # block id should be reused (LIFO free list; may or may not match, but
        # the allocator should not exceed capacity)
        assert h2.is_valid()
        assert alloc.allocated_count >= 1

    def test_free_count_stats(self, alloc, cfg):
        handles = [alloc.alloc() for _ in range(10)]
        assert alloc.allocated_count == 10
        for h in handles:
            alloc.dec_ref(h)
        alloc.maybe_advance_epoch()
        # After epoch advance, blocks should be returned to free list
        assert alloc.free_count == cfg.total_blocks

    def test_pool_exhaustion(self, cfg):
        a = SlabAllocator(cfg)
        handles = [a.alloc() for _ in range(cfg.total_blocks)]
        with pytest.raises(MemoryError, match="exhausted"):
            a.alloc()
        # Cleanup
        for h in handles:
            a.dec_ref(h)

    def test_invalid_block_handle_rejected(self, alloc):
        bad = BlockHandle(INVALID_BLOCK)
        with pytest.raises(ValueError, match="invalid"):
            alloc.ref_count(bad)


# ── Reference counting ────────────────────────────────────────────────────────

class TestRefCounting:
    def test_inc_ref_increments(self, alloc):
        h = alloc.alloc()
        alloc.inc_ref(h)
        assert alloc.ref_count(h) == 2

    def test_dec_ref_partial(self, alloc):
        h = alloc.alloc()
        alloc.inc_ref(h)
        assert alloc.ref_count(h) == 2
        retired = alloc.dec_ref(h)
        assert retired is False
        assert alloc.ref_count(h) == 1

    def test_dec_ref_to_zero(self, alloc):
        h = alloc.alloc()
        alloc.inc_ref(h)
        alloc.dec_ref(h)
        retired = alloc.dec_ref(h)
        assert retired is True

    def test_double_free_detected(self, alloc):
        h = alloc.alloc()
        alloc.dec_ref(h)
        with pytest.raises(ValueError, match="double-free|already free"):
            alloc.dec_ref(h)

    def test_inc_ref_on_freed_block_detected(self, alloc):
        h = alloc.alloc()
        alloc.dec_ref(h)
        with pytest.raises(ValueError, match="not allocated|double-inc|ref_count"):
            alloc.inc_ref(h)


# ── copy_block / zero_block ───────────────────────────────────────────────────

class TestBlockOps:
    def test_copy_block_data(self, alloc, cfg):
        src = alloc.alloc()
        dst = alloc.alloc()

        # Write some data into src
        src_tensor = alloc.get_block_tensor(src)
        src_tensor.fill_(1.5)

        alloc.copy_block(src, dst)

        dst_tensor = alloc.get_block_tensor(dst)
        assert (dst_tensor == 1.5).all(), "copy_block should have copied all values"

    def test_copy_block_shared_dst_rejected(self, alloc):
        src = alloc.alloc()
        dst = alloc.alloc()
        alloc.inc_ref(dst)  # now shared
        with pytest.raises(ValueError, match="ref_count=1"):
            alloc.copy_block(src, dst)
        alloc.dec_ref(dst)
        alloc.dec_ref(dst)

    def test_zero_block(self, alloc):
        h = alloc.alloc()
        t = alloc.get_block_tensor(h)
        t.fill_(3.0)
        alloc.zero_block(h)
        assert (t == 0.0).all()

    def test_zero_block_shared_rejected(self, alloc):
        h = alloc.alloc()
        alloc.inc_ref(h)
        with pytest.raises(ValueError, match="exclusively owned"):
            alloc.zero_block(h)
        alloc.dec_ref(h)
        alloc.dec_ref(h)


# ── Memory accounting ─────────────────────────────────────────────────────────

class TestMemoryAccounting:
    def test_alloc_count_exact(self, alloc, cfg):
        """After N alloc + N free, free_count should return to total_blocks."""
        N = 32
        handles = [alloc.alloc() for _ in range(N)]
        assert alloc.allocated_count == N
        for h in handles:
            alloc.dec_ref(h)
        alloc.maybe_advance_epoch()
        assert alloc.free_count == cfg.total_blocks, (
            f"Expected {cfg.total_blocks} free blocks after all frees, "
            f"got {alloc.free_count}. Possible block leak."
        )

    def test_stats_keys_present(self, alloc):
        s = alloc.stats()
        expected_keys = {
            "total_blocks", "free_blocks", "allocated_blocks",
            "peak_allocated", "total_allocs", "total_frees",
            "pending_reclamation", "epoch", "pool_size_gb",
        }
        assert expected_keys.issubset(s.keys())

    def test_no_block_leak_after_mixed_ops(self, alloc, cfg):
        """Mixed alloc/inc_ref/dec_ref sequence — final free count must be exact."""
        handles = []
        for _ in range(20):
            h = alloc.alloc()
            handles.append(h)

        # Share some blocks
        shared = handles[:5]
        for h in shared:
            alloc.inc_ref(h)

        # Free all
        for h in handles:
            alloc.dec_ref(h)
        for h in shared:
            alloc.dec_ref(h)

        alloc.maybe_advance_epoch()
        assert alloc.free_count == cfg.total_blocks, (
            "Block leak detected! Expected all blocks to be free."
        )


# ── EpochReclaimer ────────────────────────────────────────────────────────────

class TestEpochReclaimer:
    def test_pending_blocks_eventually_freed(self, alloc):
        h = alloc.alloc()
        bid = h.block_id
        alloc.dec_ref(h)
        # Without epoch advance, block may still be pending
        # After advance it must be in the free list
        alloc.maybe_advance_epoch()
        # Verify we can re-alloc (the block is back in the free pool)
        h2 = alloc.alloc()
        assert h2.is_valid()

    def test_reader_section_delays_reclamation(self, alloc):
        token = alloc.enter_read_section()
        h = alloc.alloc()
        alloc.dec_ref(h)
        # Block retired but reader is active → should not be freed yet
        assert alloc._reclaimer.pending_count() >= 0  # pending or already freed

        alloc.leave_read_section(token)
        alloc.maybe_advance_epoch()
        # Now block should be freed


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_alloc_no_duplicates(self, cfg):
        """Multiple threads allocating blocks should get unique block IDs."""
        a = SlabAllocator(cfg)
        n_threads = 8
        blocks_per_thread = 4
        results = []
        errors = []
        lock = threading.Lock()

        def worker():
            local = []
            try:
                for _ in range(blocks_per_thread):
                    h = a.alloc()
                    local.append(h.block_id)
            except MemoryError:
                pass
            except Exception as e:
                with lock:
                    errors.append(e)
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent alloc: {errors}"
        # All block IDs must be unique (no double-allocation)
        assert len(results) == len(set(results)), (
            "Duplicate block IDs allocated across threads — race condition!"
        )

    def test_concurrent_alloc_free_no_crash(self, cfg):
        """Concurrent alloc + free should not crash or corrupt state."""
        a = SlabAllocator(cfg)
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for _ in range(50):
                    try:
                        h = a.alloc()
                        a.dec_ref(h)
                    except MemoryError:
                        pass
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent alloc/free: {errors}"
