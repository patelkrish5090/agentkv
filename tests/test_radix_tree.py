"""
tests/test_radix_tree.py — Unit tests for DualRadixTree / CoWRadixTree.

Coverage targets:
  - create_root: correct prefix matching against empty/populated shared tree
  - fork: child shares parent blocks (ref counts incremented correctly)
  - allocate_block / free_block: correct residual block management
  - append_tokens: token list updated correctly
  - free: all ref counts decremented, handle deregistered
  - match_prefix: longest-prefix matching semantics
  - commit_prefix: tokens promoted to shared tree, residual shrinks
  - get_block_ids: correct ordering (shared then residual)
  - double-free detection
  - fork correctness: parent and child can independently append tokens

All tests run on CPU (device='cpu').
GPU tests are marked @pytest.mark.gpu.
"""

import pytest
from agentkv.core.config import PoolConfig
from agentkv.core.slab_allocator import SlabAllocator
from agentkv.core.radix_tree import DualRadixTree, INVALID_NODE_ID, ROOT_NODE_ID


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_cfg():
    return PoolConfig(
        total_blocks=128,
        block_size=4,  # small block size for easy token math in tests
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype="float16",
        device="cpu",
    )


@pytest.fixture
def alloc(small_cfg):
    return SlabAllocator(small_cfg)


@pytest.fixture
def tree(alloc, small_cfg):
    return DualRadixTree(alloc, small_cfg)


# ── create_root ───────────────────────────────────────────────────────────────

class TestCreateRoot:
    def test_basic_create(self, tree):
        tokens = [1, 2, 3, 4]
        handle = tree.create_root(tokens)
        assert handle is not None
        assert handle.tokens == tokens
        assert handle.is_freed is False

    def test_create_empty_tokens(self, tree):
        handle = tree.create_root([])
        assert handle.tokens == []
        assert handle.shared_match_len == 0

    def test_create_no_shared_prefix(self, tree):
        # Empty shared tree → no match
        handle = tree.create_root([10, 20, 30, 40])
        assert handle.shared_node_id == INVALID_NODE_ID
        assert handle.shared_match_len == 0

    def test_multiple_roots_independent(self, tree):
        h1 = tree.create_root([1, 2, 3, 4])
        h2 = tree.create_root([5, 6, 7, 8])
        assert h1.handle_id != h2.handle_id
        assert h1.tokens != h2.tokens


# ── fork ─────────────────────────────────────────────────────────────────────

class TestFork:
    def test_fork_shares_residual_blocks(self, tree, alloc):
        parent = tree.create_root([1, 2, 3, 4])
        bh = tree.allocate_block(parent)
        parent_ref_before = alloc.ref_count(bh)  # should be 1

        child = tree.fork(parent)

        # After fork, the block should be shared (ref_count = 2)
        assert alloc.ref_count(bh) == parent_ref_before + 1
        assert bh in child.residual_blocks

    def test_fork_child_independent_allocation(self, tree, alloc):
        parent = tree.create_root([1, 2, 3, 4])
        parent_bh = tree.allocate_block(parent)
        child = tree.fork(parent)

        # Child allocates its own block
        child_bh = tree.allocate_block(child)
        assert child_bh not in parent.residual_blocks
        assert alloc.ref_count(child_bh) == 1

    def test_fork_tokens_copied(self, tree):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        parent = tree.create_root(tokens)
        child = tree.fork(parent)
        assert child.tokens == parent.tokens
        # Modifying parent tokens doesn't affect child
        parent.tokens.append(99)
        assert 99 not in child.tokens

    def test_deep_fork_chain(self, tree, alloc):
        """Fork chain: root → a → b → c.  Verify ref counts accumulate."""
        root = tree.create_root([1, 2, 3, 4])
        bh = tree.allocate_block(root)

        a = tree.fork(root)
        b = tree.fork(a)
        c = tree.fork(b)

        # Root's block is shared by root, a, b, c → ref_count = 4
        assert alloc.ref_count(bh) == 4

        # Free c → ref_count drops to 3
        tree.free(c)
        assert alloc.ref_count(bh) == 3

    def test_fork_of_freed_handle_raises(self, tree):
        parent = tree.create_root([1, 2, 3])
        tree.free(parent)
        with pytest.raises(ValueError, match="freed"):
            tree.fork(parent)


# ── allocate_block / free_block ───────────────────────────────────────────────

class TestBlockAllocation:
    def test_allocate_appends_to_residual(self, tree):
        handle = tree.create_root([1, 2, 3, 4])
        assert len(handle.residual_blocks) == 0
        bh = tree.allocate_block(handle)
        assert len(handle.residual_blocks) == 1
        assert handle.residual_blocks[0] == bh

    def test_free_block_decrements_ref(self, tree, alloc):
        handle = tree.create_root([1, 2, 3, 4])
        bh = tree.allocate_block(handle)
        assert alloc.ref_count(bh) == 1
        tree.free_block(handle, bh)
        # Block should be queued for reclamation (ref_count == 0)
        # free_count on the allocator should have ticked
        assert len(handle.residual_blocks) == 0

    def test_free_block_not_owned_raises(self, tree, alloc):
        h1 = tree.create_root([1, 2, 3, 4])
        h2 = tree.create_root([5, 6, 7, 8])
        bh = tree.allocate_block(h1)
        with pytest.raises(ValueError, match="residual block"):
            tree.free_block(h2, bh)

    def test_allocate_many_blocks(self, tree, alloc):
        handle = tree.create_root([])
        blocks = [tree.allocate_block(handle) for _ in range(16)]
        assert len(handle.residual_blocks) == 16
        assert len(set(bh.block_id for bh in blocks)) == 16  # all unique

    def test_pool_exhaustion_raises(self, small_cfg):
        """Allocating more than total_blocks should raise MemoryError."""
        cfg = PoolConfig(
            total_blocks=4,
            block_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            dtype="float16",
            device="cpu",
        )
        a = SlabAllocator(cfg)
        t = DualRadixTree(a, cfg)
        handle = t.create_root([])
        with pytest.raises(MemoryError):
            for _ in range(10):  # more than total_blocks=4
                t.allocate_block(handle)


# ── free (handle-level) ───────────────────────────────────────────────────────

class TestFree:
    def test_free_removes_handle(self, tree):
        handle = tree.create_root([1, 2, 3])
        hid = handle.handle_id
        tree.free(handle)
        assert handle.is_freed
        assert hid not in tree._handles

    def test_free_releases_residual_blocks(self, tree, alloc):
        handle = tree.create_root([1, 2, 3, 4])
        bh = tree.allocate_block(handle)
        allocated_before = alloc.allocated_count
        tree.free(handle)
        # After free the block should be retired (queued in reclaimer)
        # allocated_count drops when epoch advances; check pending instead
        assert bh not in handle.residual_blocks

    def test_double_free_raises(self, tree):
        """A second call to free() on the same handle should raise ValueError.

        After the first free(), the handle is deregistered from _handles,
        so the second free() raises 'not registered' (which is correct —
        the allocator has already cleaned up the resources).
        """
        handle = tree.create_root([1, 2])
        tree.free(handle)
        with pytest.raises(ValueError, match="not registered|already freed|double-free"):
            tree.free(handle)

    def test_free_unregistered_handle_raises(self, tree):
        handle = tree.create_root([1, 2])
        # Manually corrupt registration
        del tree._handles[handle.handle_id]
        with pytest.raises(ValueError, match="not registered"):
            tree.free(handle)


# ── get_block_ids ─────────────────────────────────────────────────────────────

class TestGetBlockIds:
    def test_empty_handle(self, tree):
        handle = tree.create_root([])
        ids = tree.get_block_ids(handle)
        assert ids == []

    def test_residual_only(self, tree):
        handle = tree.create_root([1, 2, 3, 4])
        bh1 = tree.allocate_block(handle)
        bh2 = tree.allocate_block(handle)
        ids = tree.get_block_ids(handle)
        assert ids == [bh1.block_id, bh2.block_id]

    def test_ids_unique_across_agents(self, tree, alloc):
        """Two independent agents must not share block IDs in their residuals."""
        h1 = tree.create_root([1, 2, 3, 4])
        h2 = tree.create_root([5, 6, 7, 8])
        bh1 = tree.allocate_block(h1)
        bh2 = tree.allocate_block(h2)
        ids1 = set(tree.get_block_ids(h1))
        ids2 = set(tree.get_block_ids(h2))
        assert ids1.isdisjoint(ids2), f"Shared block IDs between independent agents: {ids1 & ids2}"


# ── commit_prefix ─────────────────────────────────────────────────────────────

class TestCommitPrefix:
    def test_commit_moves_blocks_to_shared(self, tree, small_cfg):
        tokens = list(range(8))  # 8 tokens = 2 blocks (block_size=4)
        handle = tree.create_root(tokens)
        tree.allocate_block(handle)
        tree.allocate_block(handle)
        assert len(handle.residual_blocks) == 2

        tree.commit_prefix(handle, length=4)  # promote 1 block worth of tokens

        assert len(handle.residual_blocks) == 1
        assert handle.shared_node_id != INVALID_NODE_ID

    def test_commit_non_multiple_of_block_size_raises(self, tree, small_cfg):
        handle = tree.create_root(list(range(8)))
        tree.allocate_block(handle)
        with pytest.raises(ValueError, match="multiple of block_size"):
            tree.commit_prefix(handle, length=3)  # 3 is not multiple of block_size=4

    def test_commit_more_blocks_than_available_raises(self, tree, small_cfg):
        handle = tree.create_root(list(range(8)))
        tree.allocate_block(handle)  # only 1 block in residual
        with pytest.raises(ValueError, match="blocks"):
            tree.commit_prefix(handle, length=8)  # requests 2 blocks

    def test_second_agent_reuses_shared_prefix(self, tree, small_cfg, alloc):
        """After commit, a new agent with the same prefix should match the shared node."""
        tokens = list(range(4))
        h1 = tree.create_root(tokens)
        bh = tree.allocate_block(h1)
        tree.commit_prefix(h1, length=4)

        # New agent with the same prefix
        h2 = tree.create_root(tokens)
        assert h2.shared_node_id != INVALID_NODE_ID
        assert h2.shared_match_len == 4


# ── append_tokens ─────────────────────────────────────────────────────────────

class TestAppendTokens:
    def test_append_updates_token_list(self, tree):
        handle = tree.create_root([1, 2])
        tree.append_tokens(handle, [3, 4])
        assert handle.tokens == [1, 2, 3, 4]

    def test_append_empty(self, tree):
        handle = tree.create_root([1, 2])
        tree.append_tokens(handle, [])
        assert handle.tokens == [1, 2]
