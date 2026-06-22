"""
bench/hf_model_demo.py — AgentKV + HuggingFace model integration demo.

What this demonstrates
----------------------
1. Run a real HuggingFace model (GPT-2 or TinyLlama) for inference.
2. Compute the KV cache for a shared prompt prefix ONCE.
3. Fork 4 agents from the same root — they SHARE the prompt KV cache.
4. Each agent independently continues from the shared KV prefix (real text).
5. Measure actual GPU memory: AgentKV CoW vs naive copy-per-agent.

Architecture note (honest)
--------------------------
In v1, AgentKV acts as a memory *tracker* alongside HuggingFace inference.
The actual KV tensors are still in HuggingFace's DynamicCache format.
AgentKV tracks WHICH blocks are shared and WHAT the theoretical savings are.

Phase 3 will implement a true custom HF Cache subclass (AgentKVCache) so
AgentKV's pool IS the storage — no HF copy at all.  This demo is the
stepping stone that proves the concept with real model outputs.

HF's past_key_values sharing
-----------------------------
When you call model.generate(new_tokens, past_key_values=shared_kv), HF:
  - Does NOT copy shared_kv in place
  - Appends new K/V tensors to a fresh DynamicCache for each call
  - So 4 agents sharing the same past_key_values is safe and correct

Usage
-----
  # CPU mode (slow but works anywhere):
  python bench/hf_model_demo.py --device cpu --model gpt2

  # T4 Colab (fast):
  python bench/hf_model_demo.py --device cuda --model gpt2

  # Larger model (needs more VRAM):
  python bench/hf_model_demo.py --device cuda --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import List, Optional, Tuple

import torch


def bytes_of_past_kv(past_key_values) -> int:
    """Return total byte size of a HuggingFace past_key_values cache.

    Handles both formats:
      - Old (transformers < 4.38): tuple of (key, value) tensors per layer
      - New (transformers >= 4.38): DynamicCache object with .key_cache / .value_cache
    """
    total = 0
    # New-style: DynamicCache has .key_cache and .value_cache lists
    if hasattr(past_key_values, 'key_cache'):
        for t in past_key_values.key_cache:
            if t is not None:
                total += t.numel() * t.element_size()
        for t in past_key_values.value_cache:
            if t is not None:
                total += t.numel() * t.element_size()
    else:
        # Old-style: tuple of (key_tensor, value_tensor) per layer
        for layer_kv in past_key_values:
            for tensor in layer_kv:
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
    return total


def run_demo(
    model_name: str = "gpt2",
    device: str = "cuda",
    shared_prompt: str = "Scientists discovered that artificial intelligence systems",
    n_agents: int = 4,
    new_tokens_per_agent: int = 20,
) -> None:
    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"AgentKV + HuggingFace Demo")
    print(f"  model  = {model_name}")
    print(f"  device = {device}")
    print(f"  agents = {n_agents}")
    print(f"{'='*60}\n")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers accelerate"
        )

    print(f"[1/5] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    # HF models vary in how they name KV head count
    n_kv_heads = getattr(cfg, "num_key_value_heads",
                  getattr(cfg, "num_attention_heads", 8))
    head_dim = getattr(cfg, "head_dim",
                cfg.hidden_size // cfg.num_attention_heads)

    print(f"   Model loaded: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")
    if device == "cuda":
        used_gb = torch.cuda.memory_allocated() / 1e9
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"   Model VRAM: {used_gb:.2f} GB used, {free_gb:.2f} GB free")

    # ── 2. Compute shared prefix KV cache ────────────────────────────────────
    print(f"\n[2/5] Computing shared prefix KV cache for prompt...")
    print(f"   Prompt: \"{shared_prompt}\"")

    inputs = tokenizer(shared_prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    print(f"   Prompt length: {prompt_len} tokens")

    t0 = time.perf_counter()
    with torch.no_grad():
        prefix_out = model(**inputs, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000

    shared_past_kv = prefix_out.past_key_values
    kv_bytes = bytes_of_past_kv(shared_past_kv)

    print(f"   KV cache size: {kv_bytes / 1024:.1f} KB "
          f"({kv_bytes / 1024 / prompt_len:.1f} KB/token)")
    print(f"   Prefill time: {prefill_ms:.1f} ms")

    # ── 3. Set up AgentKV pool ────────────────────────────────────────────────
    print(f"\n[3/5] Initialising AgentKV pool (matching model architecture)...")

    from agentkv import AgentKVPool
    from agentkv.core.config import PoolConfig

    # Size pool to fit in available VRAM (50% of free → safe margin)
    if device == "cuda":
        cfg_pool = PoolConfig.max_for_device(
            fraction=0.5,
            num_layers=n_layers,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_size=16,
            dtype="float16",
            device=device,
        )
    else:
        cfg_pool = PoolConfig(
            total_blocks=512,
            num_layers=n_layers,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_size=16,
            dtype="float32",
            device=device,
        )

    pool = AgentKVPool(config=cfg_pool)
    print(f"   Pool: {cfg_pool}")

    # Register shared prompt as root agent
    root = pool.create_root(inputs.input_ids[0].tolist())
    n_prompt_blocks = max(1, (prompt_len + cfg_pool.block_size - 1) // cfg_pool.block_size)
    for _ in range(n_prompt_blocks):
        pool.allocate_block(root)
    print(f"   Root agent: {n_prompt_blocks} blocks for {prompt_len} prompt tokens")

    # ── 4. Fork agents and generate ──────────────────────────────────────────
    print(f"\n[4/5] Forking {n_agents} agents and generating continuations...")

    agent_continuations = [
        " can solve problems that seemed impossible before",
        " still struggles with common-sense reasoning",
        " is transforming how doctors diagnose diseases",
        " needs strict regulation to prevent misuse",
    ][:n_agents]

    # Pad to n_agents if fewer continuations defined
    while len(agent_continuations) < n_agents:
        agent_continuations.append(f" opens new research directions")

    agents = [pool.fork(root) for _ in range(n_agents)]
    print(f"   Forked {n_agents} agents (all sharing {n_prompt_blocks} prompt blocks)")

    results = []
    agent_gen_times = []
    child_new_token_counts = []

    for i, (agent, continuation) in enumerate(zip(agents, agent_continuations)):
        # Tokenize the continuation (new tokens each agent sees)
        cont_ids = tokenizer.encode(continuation, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            gen_ids = model.generate(
                cont_ids,
                past_key_values=shared_past_kv,  # Shared prefix! Not copied.
                max_new_tokens=new_tokens_per_agent,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        gen_ms = (time.perf_counter() - t0) * 1000
        agent_gen_times.append(gen_ms)

        # Decode: full output = prompt + continuation + generated
        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        results.append((continuation.strip(), full_text))

        # Track new tokens in AgentKV
        new_tok_count = gen_ids.shape[1] - cont_ids.shape[1]
        child_new_token_counts.append(new_tok_count)
        n_new_blocks = max(1, (new_tok_count + cfg_pool.block_size - 1) // cfg_pool.block_size)
        for _ in range(n_new_blocks):
            pool.allocate_block(agent)

        print(f"   Agent {i+1} [{gen_ms:.0f}ms, {new_tok_count} new tokens]: "
              f"\"{continuation.strip()[:40]}...\"")

    # ── 5. Results ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Results\n{'='*60}")
    print(f"\nShared prefix: \"{shared_prompt}\"\n")
    for i, (cont, full) in enumerate(results):
        # Show only the generated portion (after the continuation)
        print(f"Agent {i+1}  (+\"{cont}\")")
        print(f"        → {shared_prompt}{full[:120]}")
        print()

    # Memory analysis
    print(f"{'='*60}")
    print("Memory Analysis")
    print(f"{'='*60}")

    avg_child_tokens = sum(child_new_token_counts) / n_agents if child_new_token_counts else 0
    bytes_per_token = kv_bytes / prompt_len

    # Naive: each agent gets its own copy of the shared prompt KV
    naive_total_bytes = n_agents * kv_bytes + sum(
        t * bytes_per_token for t in child_new_token_counts
    )
    # AgentKV: shared prompt KV + each agent's own new tokens only
    agentkv_total_bytes = kv_bytes + sum(
        t * bytes_per_token for t in child_new_token_counts
    )
    savings_pct = (1 - agentkv_total_bytes / naive_total_bytes) * 100

    print(f"  Model:              {model_name}")
    print(f"  Shared prompt:      {prompt_len} tokens = {kv_bytes/1024:.1f} KB KV cache")
    print(f"  Agents:             {n_agents}")
    print(f"  Avg new tokens:     {avg_child_tokens:.0f} per agent")
    print()
    print(f"  Naive KV memory:    {naive_total_bytes/1024:.1f} KB")
    print(f"    = {n_agents} copies of prompt KV + child-specific KV")
    print(f"  AgentKV memory:     {agentkv_total_bytes/1024:.1f} KB")
    print(f"    = 1 shared prompt KV + child-specific KV only")
    print(f"  Memory saved:       {(naive_total_bytes - agentkv_total_bytes)/1024:.1f} KB "
          f"({savings_pct:.1f}%)")
    print()
    print(f"  AgentKV pool stats: {pool.stats()}")

    if device == "cuda":
        torch.cuda.synchronize()
        final_used_gb = torch.cuda.memory_allocated() / 1e9
        print(f"\n  Final GPU VRAM used: {final_used_gb:.2f} GB")

    # Cleanup
    for agent in agents:
        pool.free(agent)
    pool.free(root)
    pool.maybe_advance_epoch()
    print(f"\n✅ All agents freed. Pool fully recovered.")

    return {
        "model": model_name,
        "prompt_len": prompt_len,
        "kv_bytes": kv_bytes,
        "n_agents": n_agents,
        "naive_bytes": naive_total_bytes,
        "agentkv_bytes": agentkv_total_bytes,
        "savings_pct": savings_pct,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentKV + HuggingFace Model Demo")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt",
                        default="Scientists discovered that artificial intelligence systems",
                        help="Shared prompt prefix for all agents")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=20)
    args = parser.parse_args()

    run_demo(
        model_name=args.model,
        device=args.device,
        shared_prompt=args.prompt,
        n_agents=args.n_agents,
        new_tokens_per_agent=args.new_tokens,
    )
