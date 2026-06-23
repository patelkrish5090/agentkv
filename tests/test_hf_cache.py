import pytest
import torch

try:
    import transformers
except ImportError:
    pytest.skip("transformers not installed", allow_module_level=True)

from agentkv import AgentKVPool, PoolConfig
from agentkv.hf_cache import AgentKVCache

@pytest.fixture
def pool():
    cfg = PoolConfig(
        total_blocks=16,
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        dtype="float32",
        device="cpu",
    )
    return AgentKVPool(config=cfg)

def test_hf_cache_update_and_get(pool):
    # 1. Create root agent
    prompt_tokens = [101, 102, 103, 104, 105]  # len = 5
    root_handle = pool.create_root(prompt_tokens)
    
    # 2. Init cache
    cache = AgentKVCache(pool, root_handle)
    assert cache.get_seq_length() == 5
    
    # 3. Simulate prefill of the 5 tokens
    # shape: [1, num_heads, seq_len, head_dim]
    num_heads = pool.config.num_kv_heads
    head_dim = pool.config.head_dim
    
    k_states_0 = torch.randn(1, num_heads, 5, head_dim)
    v_states_0 = torch.randn(1, num_heads, 5, head_dim)
    k_states_1 = torch.randn(1, num_heads, 5, head_dim)
    v_states_1 = torch.randn(1, num_heads, 5, head_dim)
    
    # layer 0
    cache._seen_tokens = 0 # reset to 0 to simulate prefilling the entire prompt
    out_k0, out_v0 = cache.update(k_states_0, v_states_0, 0)
    assert out_k0.shape == (1, num_heads, 5, head_dim)
    assert torch.allclose(out_k0, k_states_0)
    
    # layer 1
    out_k1, out_v1 = cache.update(k_states_1, v_states_1, 1)
    assert out_k1.shape == (1, num_heads, 5, head_dim)
    assert torch.allclose(out_k1, k_states_1)
    
    # 4. Check sequence length
    assert cache.get_seq_length() == 5
    
    # 5. Check block allocation
    # 5 tokens with block_size=4 needs 2 blocks
    block_ids = pool.get_block_ids(root_handle)
    assert len(block_ids) == 2

def test_hf_cache_fork_and_cow(pool):
    # Pre-populate root
    prompt_tokens = [101, 102, 103, 104]  # Exactly 1 block
    root_handle = pool.create_root(prompt_tokens)
    cache = AgentKVCache(pool, root_handle)
    cache._seen_tokens = 0
    
    num_heads = pool.config.num_kv_heads
    head_dim = pool.config.head_dim
    k_states = torch.ones(1, num_heads, 4, head_dim)
    v_states = torch.ones(1, num_heads, 4, head_dim)
    
    cache.update(k_states, v_states, 0)
    cache.update(k_states, v_states, 1)
    
    # Commit prefix to share
    pool.commit_prefix(root_handle, 4)
    
    # Fork two agents
    child1_cache = cache.fork()
    child2_cache = cache.fork()
    
    assert child1_cache.get_seq_length() == 4
    assert child2_cache.get_seq_length() == 4
    
    # Generate 1 token for child 1
    k_new_1 = torch.full((1, num_heads, 1, head_dim), 2.0)
    v_new_1 = torch.full((1, num_heads, 1, head_dim), 2.0)
    
    # It should allocate a new block for child 1 (since block 1 is full)
    out_k_c1_0, _ = child1_cache.update(k_new_1, v_new_1, 0)
    child1_cache.update(k_new_1, v_new_1, 1)
    
    assert child1_cache.get_seq_length() == 5
    assert out_k_c1_0.shape == (1, num_heads, 5, head_dim)
    # The first 4 tokens should be 1.0, the last token should be 2.0
    assert torch.allclose(out_k_c1_0[:, :, :4, :], torch.ones(1, num_heads, 4, head_dim))
    assert torch.allclose(out_k_c1_0[:, :, 4:, :], torch.full((1, num_heads, 1, head_dim), 2.0))
    
    # Ensure child2 is unaffected
    assert child2_cache.get_seq_length() == 4
    
    # Get legacy format for child 1 to verify export
    legacy = child1_cache.to_legacy_cache()
    assert len(legacy) == 2  # num_layers
    assert legacy[0][0].shape == (1, num_heads, 5, head_dim)
