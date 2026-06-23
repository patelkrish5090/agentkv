"""
bench/hf_cache_demo.py — End-to-end HuggingFace demo with real GPU CoW.

Unlike `hf_model_demo.py` (which just tracked allocations in metadata and used
copy.deepcopy() for the actual HF cache), this script passes an `AgentKVCache`
to the model. The model reads and writes directly from/to the AgentKVPool's
slab tensor using real zero-copy forks!
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentkv import AgentKVPool, PoolConfig
from agentkv.hf_cache import AgentKVCache

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("AgentKV Phase 3a — Real GPU CoW Demo")
    print(f"  model      = {args.model}")
    print(f"  device     = {args.device}")
    print(f"  agents     = {args.n_agents}")
    print(f"  new_tokens = {args.new_tokens}")
    print("=" * 60)

    # ── 1. Load Model ──
    print(f"\n[1/5] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        device_map=args.device,
    )
    
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    num_kv_heads = getattr(cfg, "num_key_value_heads", getattr(cfg, "n_head", 0))
    head_dim = getattr(cfg, "head_dim", getattr(cfg, "n_embd", 0) // getattr(cfg, "n_head", 1))

    # ── 2. Initialize AgentKVPool ──
    print("\n[2/5] Initializing AgentKV Pool...")
    pool_cfg = PoolConfig(
        total_blocks=256,
        block_size=16,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype="float32" if args.device == "cpu" else "float16",
        device=args.device,
    )
    pool = AgentKVPool(config=pool_cfg)
    print(pool)

    # ── 3. Shared Prefix Prefill ──
    print("\n[3/5] Prefilling shared system prompt...")
    system_prompt = (
        "You are a helpful AI assistant evaluating a copy-on-write memory allocator. "
        "The following is a creative writing task. Continue the story naturally.\n\n"
        "Story: The ancient library had been sealed for centuries, guarded by complex "
        "mechanisms. When the explorers finally breached the inner sanctum, they found "
    )
    inputs = tokenizer(system_prompt, return_tensors="pt").to(args.device)
    prompt_ids = inputs.input_ids[0].tolist()

    # Create root agent and the associated HF cache
    root_handle = pool.create_root(prompt_ids)
    root_cache = AgentKVCache(pool, root_handle)
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        prefix_out = model(
            **inputs,
            past_key_values=root_cache,
            use_cache=True,
        )
    prefill_time = time.perf_counter() - t0
    
    # We must explicitly commit the prefix so it's placed in the shared tree
    # Round down to block boundary for safe sharing
    safe_share_len = (len(prompt_ids) // pool.block_size) * pool.block_size
    pool.commit_prefix(root_handle, safe_share_len)
    
    print(f"   Tokens : {len(prompt_ids)}")
    print(f"   Time   : {prefill_time * 1000:.1f} ms")
    print(f"   {pool}")

    # ── 4. Branching (GPU CoW) ──
    print("\n[4/5] Forking agents and generating...")
    agent_caches = []
    
    # Notice: NO copy.deepcopy()! We use real CoW forks.
    for i in range(args.n_agents):
        child_cache = root_cache.fork()
        agent_caches.append(child_cache)
        
    print(f"   {pool}")

    agent_texts = ["" for _ in range(args.n_agents)]
    agent_input_ids = [inputs.input_ids.clone() for _ in range(args.n_agents)]
    
    # To ensure different outputs despite the same prompt (since greedy decoding is deterministic),
    # we inject a unique first token for each agent.
    continuations = [
        "a glowing ",
        "nothing but ",
        "thousands of ",
        "a single ",
        "an empty "
    ]
    
    for i in range(args.n_agents):
        cont_str = continuations[i % len(continuations)]
        cont_ids = tokenizer(cont_str, return_tensors="pt").to(args.device).input_ids
        
        # We append the first continuation tokens one by one
        # Note: Since the cache already has the prompt, we only pass the new tokens.
        # But we must update the cache correctly. HF `generate` does this automatically.
        # Let's just use `model.generate` with the agent's cache!
        
        full_ids = torch.cat([agent_input_ids[i], cont_ids], dim=1)
        
        gen_ids = model.generate(
            full_ids,
            past_key_values=agent_caches[i],
            max_new_tokens=args.new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        agent_texts[i] = tokenizer.decode(gen_ids[0][len(prompt_ids):], skip_special_tokens=True)
        print(f"\n   Agent {i+1} output: {agent_texts[i]}")

    # ── 5. Stats ──
    print("\n[5/5] Final Stats")
    print(f"   {pool}")
    
    # Calculation
    blocks_per_agent = (len(prompt_ids) + args.new_tokens + pool.block_size - 1) // pool.block_size
    naive_blocks = blocks_per_agent * args.n_agents
    
    actual_blocks = pool.allocated_blocks
    savings = 100.0 * (1.0 - (actual_blocks / naive_blocks))
    
    print(f"\nMemory Efficiency:")
    print(f"   Naive independent allocation: {naive_blocks} blocks")
    print(f"   AgentKV real allocation   : {actual_blocks} blocks")
    print(f"   GPU Memory Saved          : {savings:.1f}%")

if __name__ == "__main__":
    main()
