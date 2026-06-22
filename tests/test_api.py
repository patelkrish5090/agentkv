"""
tests/test_api.py — Tests for the AgentKVPool public API (Phase 2).

These tests verify:
  1. The 10-line quickstart script from the README runs end-to-end.
  2. All public methods work correctly through the pool interface.
  3. Stats reporting is accurate.
  4. The config factory (from_capacity_gb) works correctly.
"""

import pytest
from agentkv import AgentKVPool
from agentkv.core.config import PoolConfig


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_pool():
    """A tiny pool for fast API tests (CPU, minimal memory)."""
    cfg = PoolConfig(
        total_blocks=128,
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype="float16",
        device="cpu",
    )
    return AgentKVPool(config=cfg)


# ── Quickstart smoke test ─────────────────────────────────────────────────────

class TestQuickstart:
    def test_10_line_script(self, small_pool):
        """The README 10-line quickstart must execute without error."""
        pool = small_pool

        root = pool.create_root([1, 2, 3, 4])
        child_a = pool.fork(root)
        child_b = pool.fork(root)

        pool.allocate_block(child_a)
        pool.allocate_block(child_b)
        pool.allocate_block(root)

        stats = pool.stats()
        assert stats["active_handles"] == 3
        assert stats["allocated_blocks"] > 0

        pool.free(child_a)
        pool.free(child_b)
        pool.free(root)

        assert pool.stats()["active_handles"] == 0

    def test_repr_works(self, small_pool):
        r = repr(small_pool)
        assert "AgentKVPool" in r
        assert "GiB" in r


# ── Pool config factory ───────────────────────────────────────────────────────

class TestPoolConfig:
    def test_from_capacity_gb_creates_valid_pool(self):
        pool = AgentKVPool(
            capacity_gb=0.001,  # tiny budget for testing
            block_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            dtype="float16",
            device="cpu",
        )
        assert pool.block_size == 4
        assert pool.config.total_blocks >= 1

    def test_explicit_config_overrides_params(self):
        cfg = PoolConfig(
            total_blocks=64,
            block_size=8,
            num_layers=1,
            num_kv_heads=1,
            head_dim=8,
            dtype="float32",
            device="cpu",
        )
        pool = AgentKVPool(config=cfg)
        assert pool.block_size == 8
        assert pool.config.dtype == "float32"


# ── create_root / fork / free ─────────────────────────────────────────────────

class TestPoolAPI:
    def test_create_and_free(self, small_pool):
        h = small_pool.create_root([10, 20, 30, 40])
        assert small_pool.stats()["active_handles"] == 1
        small_pool.free(h)
        assert small_pool.stats()["active_handles"] == 0

    def test_fork_increases_allocated(self, small_pool):
        root = small_pool.create_root([1, 2, 3, 4])
        small_pool.allocate_block(root)
        alloc_before = small_pool.allocated_blocks

        child = small_pool.fork(root)
        # Forking increments ref count — allocated_count doesn't change
        # (ref count goes from 1 → 2 but block count stays the same)
        assert small_pool.allocated_blocks == alloc_before

        small_pool.free(child)
        small_pool.free(root)

    def test_fork_child_gets_own_blocks(self, small_pool):
        root = small_pool.create_root([1, 2, 3, 4])
        child = small_pool.fork(root)

        small_pool.allocate_block(child)
        child_ids = set(small_pool.get_block_ids(child))
        root_ids = set(small_pool.get_block_ids(root))

        # Child's new block should not be in root's block list
        new_block_ids = child_ids - root_ids
        assert len(new_block_ids) >= 1

        small_pool.free(child)
        small_pool.free(root)

    def test_get_block_ids_is_list_of_ints(self, small_pool):
        h = small_pool.create_root([1, 2, 3, 4])
        small_pool.allocate_block(h)
        ids = small_pool.get_block_ids(h)
        assert isinstance(ids, list)
        assert all(isinstance(x, int) for x in ids)
        small_pool.free(h)

    def test_append_tokens_updates_state(self, small_pool):
        h = small_pool.create_root([1, 2])
        small_pool.append_tokens(h, [3, 4])
        assert h.tokens == [1, 2, 3, 4]
        small_pool.free(h)

    def test_free_count_property(self, small_pool):
        initial = small_pool.free_blocks
        h = small_pool.create_root([])
        small_pool.allocate_block(h)
        assert small_pool.free_blocks == initial - 1
        small_pool.free(h)


# ── Stats ─────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_keys(self, small_pool):
        stats = small_pool.stats()
        for key in ["total_blocks", "free_blocks", "allocated_blocks",
                    "active_handles", "shared_nodes", "pool_size_gb"]:
            assert key in stats, f"Missing stats key: {key}"

    def test_stats_after_fork_tree(self, small_pool):
        root = small_pool.create_root([1, 2, 3, 4])
        children = [small_pool.fork(root) for _ in range(4)]
        assert small_pool.stats()["active_handles"] == 5  # root + 4 children
        for c in children:
            small_pool.free(c)
        small_pool.free(root)
        assert small_pool.stats()["active_handles"] == 0


# ── GPU quickstart ─────────────────────────────────────────────────────────────

@pytest.mark.gpu
class TestGPUAPI:
    def test_gpu_create_fork_free(self):
        cfg = PoolConfig(
            total_blocks=256,
            block_size=16,
            num_layers=4,
            num_kv_heads=4,
            head_dim=64,
            dtype="float16",
            device="cuda",
        )
        pool = AgentKVPool(config=cfg)
        root = pool.create_root(list(range(16)))
        child = pool.fork(root)
        pool.allocate_block(child)
        assert pool.stats()["allocated_blocks"] == 1
        pool.free(child)
        pool.free(root)
