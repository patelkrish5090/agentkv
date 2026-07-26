"""
tests/test_vllm_integration.py — Tests for AgentKVBlockManager (integrations/vllm).

Real vLLM cannot be installed in this dev environment (Triton + CUDA build
required, not available on native Windows). These tests instead run the
integration against a minimal stub of vllm.core.interfaces / vllm.sequence
in tests/fakes/, whose signatures were copied from the vendored reference
in vllm-src/. This validates AgentKVBlockManager's actual logic — prefix
sharing, fork, CoW, free — without needing a real vLLM install or a GPU.

If real vLLM is ever installed in this environment, it will already be in
sys.modules by the time these tests run and takes precedence over the fake.
"""

import os
import sys

import pytest

_FAKES_DIR = os.path.join(os.path.dirname(__file__), "fakes")
if _FAKES_DIR not in sys.path:
    sys.path.insert(0, _FAKES_DIR)

from vllm.sequence import Sequence, SequenceGroup, SequenceStatus  # noqa: E402

from integrations.vllm.block_manager import AgentKVBlockManager  # noqa: E402

pytestmark = pytest.mark.integration

BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 256


@pytest.fixture
def manager() -> AgentKVBlockManager:
    return AgentKVBlockManager(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_GPU_BLOCKS,
        num_cpu_blocks=0,
    )


def _seq_group(seq_id: int, token_ids: list[int]) -> SequenceGroup:
    return SequenceGroup([Sequence(seq_id, token_ids)])


def test_allocate_single_sequence(manager: AgentKVBlockManager):
    tokens = list(range(40))  # 3 blocks at block_size=16
    group = _seq_group(1, tokens)

    assert manager.can_allocate(group) is not None
    manager.allocate(group)

    seq = group.get_seqs()[0]
    block_table = manager.get_block_table(seq)
    assert len(block_table) == (len(tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE


def test_prefix_sharing_across_requests(manager: AgentKVBlockManager):
    """Second request with the same prefix should reuse committed blocks
    instead of raising (this used to crash — commit_prefix() was being
    called with the absolute prompt length instead of the residual length
    beyond the already-matched shared prefix)."""
    shared_prefix = list(range(32))  # 2 full blocks, will be committed

    group_a = _seq_group(1, shared_prefix)
    manager.allocate(group_a)
    free_after_a = manager.get_num_free_gpu_blocks()

    # Second request: same 2-block prefix + one new block of tokens.
    group_b = _seq_group(2, shared_prefix + list(range(1000, 1016)))
    manager.allocate(group_b)  # must not raise

    seq_b = group_b.get_seqs()[0]
    block_table_b = manager.get_block_table(seq_b)
    assert len(block_table_b) == 3  # 2 shared + 1 new

    # The 2 shared blocks must not have been re-allocated from the free pool.
    free_after_b = manager.get_num_free_gpu_blocks()
    assert free_after_a - free_after_b == 1


def test_fork_for_beam_search(manager: AgentKVBlockManager):
    tokens = list(range(16))
    parent = Sequence(1, tokens)
    child = Sequence(2, tokens)
    group = SequenceGroup([parent, child])

    manager.allocate(group)  # allocate() forks waiting_seqs[1:] from seq[0]

    parent_blocks = manager.get_block_table(parent)
    child_blocks = manager.get_block_table(child)
    assert parent_blocks == child_blocks  # shares blocks right after fork


def test_append_slots_triggers_cow_on_shared_block(manager: AgentKVBlockManager):
    # 20 tokens = 1 full block (committed to the shared tree) + 1 partially
    # filled residual block (4/16 tokens used). The partial block is still a
    # *residual* block at fork time, so fork() shares it via ref-count — this
    # is the one case where two agents can end up writing into the same
    # physical block and a real CoW copy is required.
    tokens = list(range(20))
    parent = Sequence(1, tokens)
    child = Sequence(2, tokens)
    group = SequenceGroup([parent, child])
    manager.allocate(group)

    parent_handle = manager.seq_to_handle[parent.seq_id]
    child_handle = manager.seq_to_handle[child.seq_id]
    assert parent_handle.residual_blocks == child_handle.residual_blocks
    shared_residual_block = parent_handle.residual_blocks[0]
    assert manager.pool._allocator.ref_count(shared_residual_block) == 2

    # Parent fills the next slot in that shared partial block -> must CoW.
    parent.append_token_id(9001)
    cows_parent = manager.append_slots(parent, num_lookahead_slots=0)
    assert cows_parent, "expected a CoW copy when writing into a ref-counted residual block"

    # Parent no longer shares physical storage with the child for that block.
    assert manager.pool._allocator.ref_count(shared_residual_block) == 1
    assert parent_handle.residual_blocks[0] != child_handle.residual_blocks[0]

    # Child is now the sole owner of the original block, so its own append
    # does not need to copy anything.
    child.append_token_id(9002)
    cows_child = manager.append_slots(child, num_lookahead_slots=0)
    assert cows_child == []


def test_free_releases_blocks(manager: AgentKVBlockManager):
    tokens = list(range(32))
    group = _seq_group(1, tokens)
    manager.allocate(group)
    seq = group.get_seqs()[0]

    free_before = manager.get_num_free_gpu_blocks()
    manager.free(seq)
    manager.pool.maybe_advance_epoch()
    free_after = manager.get_num_free_gpu_blocks()

    assert free_after > free_before
    assert seq.seq_id not in manager.seq_to_handle
