"""
bench/hf_model_demo.py — AgentKV + HuggingFace model integration demo.

What this demonstrates
----------------------
1. Run a real HuggingFace model (GPT-2) for inference.
2. Compute the KV cache for a shared prompt prefix ONCE.
3. Fork 4 agents from the same root — they SHARE the prompt KV via AgentKV CoW.
4. Each agent independently continues from the shared KV prefix (real text).
5. Measure actual GPU memory: AgentKV CoW vs naive copy-per-agent.

Architecture note (honest)
--------------------------
In v1, AgentKV acts as a memory *tracker* alongside HuggingFace inference.
The actual KV tensors live in HuggingFace's DynamicCache format.
AgentKV tracks WHICH blocks are shared and WHAT the theoretical savings are.

Phase 3 will implement a true custom HF Cache subclass (AgentKVCache) so
AgentKV's slab pool IS the storage — no HF copy overhead at all.

HF past_key_values API changes
-------------------------------
transformers < 4.38  : model() returns tuple of (K,V) tuples; generate() accepts tuples.
transformers 4.38-4.40: model() returns DynamicCache OR tuple depending on model.
transformers >= 4.41 : generate() HARD-REJECTS tuples — must be a Cache instance.

Fix: call to_dynamic_cache() right after model() to normalise before any generate() call.

Usage
-----
  python bench/hf_model_demo.py --device cuda --model gpt2
  python bench/hf_model_demo.py --device cuda --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import time

import torch


# ── HF Cache helpers ──────────────────────────────────────────────────────────
#
# DynamicCache internal attributes (key_cache, value_cache) have shifted
# between transformers versions.  We use ONLY the stable public API:
#   - DynamicCache.update(k, v, layer_idx)  — to build / populate
#   - DynamicCache.to_legacy_cache()        — to iterate for byte counting
#   - copy.deepcopy()                       — to clone

import copy


def bytes_of_cache(cache) -> int:
    """Return the total byte size of a HF cache (any version)."""
    total = 0
    try:
        # DynamicCache: to_legacy_cache() → tuple of (K, V) per layer
        for k, v in cache.to_legacy_cache():
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
    except Exception:
        # Fallback: raw tuple-of-tuples from very old or very new models
        for layer_kv in cache:
            for t in layer_kv:
                if t is not None:
                    total += t.numel() * t.element_size()
    return total


def to_dynamic_cache(past_key_values):
    """Normalise any past_key_values to a DynamicCache.

    Called IMMEDIATELY after model() forward — before any generate() call.
    transformers >= 4.41 hard-rejects tuples in generate(); we fix that here.
    """
    from transformers import DynamicCache

    # Already correct — nothing to do
    if isinstance(past_key_values, DynamicCache):
        return past_key_values

    # Try the official helper introduced in transformers 4.44
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past_key_values)

    # Manual construction via the stable update() public API
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_key_values):
        cache.update(k, v, layer_idx)
    return cache


def clone_dynamic_cache(cache):
    """Deep-clone a DynamicCache so each agent gets its own mutable copy.

    generate() extends the cache in-place per token; agents must not share
    a live cache or they corrupt each other's decode state.

    In Phase 3's AgentKVCache this is replaced by a GPU CoW fork — zero-copy
    until the first write — but copy.deepcopy() is correct here for any
    transformers version regardless of internal attribute naming.
    """
    return copy.deepcopy(cache)


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo(
    model_name: str = "gpt2",
    device: str = "cuda",
    shared_prompt: str = "Scientists discovered that artificial intelligence systems",
    n_agents: int = 4,
    new_tokens_per_agent: int = 20,
) -> dict:

    print(f"\n{'='*60}")
    print(f"AgentKV + HuggingFace Demo")
    print(f"  model  = {model_name}")
    print(f"  device = {device}")
    print(f"  agents = {n_agents}")
    print(f"{'='*60}\n")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("pip install transformers accelerate")

    print(f"[1/5] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    mcfg = model.config
    n_layers   = mcfg.num_hidden_layers
    n_kv_heads = getattr(mcfg, "num_key_value_heads",
                  getattr(mcfg, "num_attention_heads", 8))
    head_dim   = getattr(mcfg, "head_dim",
                  mcfg.hidden_size // mcfg.num_attention_heads)

    print(f"   {n_layers} layers | {n_kv_heads} KV heads | head_dim={head_dim}")
    if device == "cuda":
        print(f"   VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB used, "
              f"{torch.cuda.mem_get_info()[0]/1e9:.2f} GB free")

    # ── 2. Prefill shared prompt → get KV cache ───────────────────────────────
    print(f"\n[2/5] Prefilling shared prompt...")
    print(f"   \"{shared_prompt}\"")

    inputs     = tokenizer(shared_prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    print(f"   {prompt_len} tokens")

    t0 = time.perf_counter()
    with torch.no_grad():
        prefix_out = model(**inputs, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Normalise to DynamicCache IMMEDIATELY — this is the key fix.
    # GPT-2's model() returns a legacy tuple; generate() >= 4.41 rejects tuples.
    shared_cache = to_dynamic_cache(prefix_out.past_key_values)
    kv_bytes     = bytes_of_cache(shared_cache)

    print(f"   KV cache: {kv_bytes/1024:.1f} KB  |  prefill: {prefill_ms:.0f} ms")

    # ── 3. AgentKV pool (sized to model architecture) ─────────────────────────
    print(f"\n[3/5] Initialising AgentKV pool...")

    from agentkv import AgentKVPool
    from agentkv.core.config import PoolConfig

    if device == "cuda":
        pool_cfg = PoolConfig.max_for_device(
            fraction=0.5,
            num_layers=n_layers,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_size=16,
            dtype="float16",
            device=device,
        )
    else:
        pool_cfg = PoolConfig(
            total_blocks=512,
            num_layers=n_layers,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_size=16,
            dtype="float32",
            device=device,
        )

    pool            = AgentKVPool(config=pool_cfg)
    root            = pool.create_root(inputs.input_ids[0].tolist())
    n_prompt_blocks = max(1, (prompt_len + pool_cfg.block_size - 1) // pool_cfg.block_size)
    for _ in range(n_prompt_blocks):
        pool.allocate_block(root)

    print(f"   {pool_cfg}")
    print(f"   Root: {n_prompt_blocks} block(s) for {prompt_len} prompt tokens")

    # ── 4. Fork agents and run generation ────────────────────────────────────
    print(f"\n[4/5] Forking {n_agents} agents + generating...")

    continuations = [
        " can solve problems that seemed impossible before",
        " still struggles with common-sense reasoning",
        " is transforming how doctors diagnose diseases",
        " needs strict regulation to prevent misuse",
        " will reshape education in fundamental ways",
        " creates new opportunities in every industry",
        " raises deep questions about human creativity",
        " depends on the quality of training data",
    ][:n_agents]
    while len(continuations) < n_agents:
        continuations.append(" opens many new research directions")

    agents = [pool.fork(root) for _ in range(n_agents)]
    print(f"   {n_agents} agents forked — all share {n_prompt_blocks} prompt block(s)")

    results              = []
    child_new_tok_counts = []

    for i, (agent, cont) in enumerate(zip(agents, continuations)):
        cont_ids = tokenizer.encode(cont, return_tensors="pt").to(device)

        # Clone the shared DynamicCache for this agent.
        # generate() appends new KVs in place, so each agent needs its own copy.
        # (In Phase 3's AgentKVCache this becomes a zero-copy GPU CoW fork.)
        t0 = time.perf_counter()
        agent_cache = clone_dynamic_cache(shared_cache)

        with torch.no_grad():
            gen_ids = model.generate(
                cont_ids,
                past_key_values=agent_cache,   # DynamicCache — always accepted
                max_new_tokens=new_tokens_per_agent,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        gen_ms = (time.perf_counter() - t0) * 1000

        full_text    = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        new_tok_count = gen_ids.shape[1] - cont_ids.shape[1]
        results.append((cont.strip(), full_text))
        child_new_tok_counts.append(new_tok_count)

        # Track new blocks in AgentKV
        n_new_blocks = max(1, (new_tok_count + pool_cfg.block_size - 1) // pool_cfg.block_size)
        for _ in range(n_new_blocks):
            pool.allocate_block(agent)

        print(f"   Agent {i+1} [{gen_ms:.0f} ms | {new_tok_count} new tokens]: "
              f"\"{cont.strip()[:45]}...\"")

    # ── 5. Results ────────────────────────────────────────────────────────────
    print(f"\n[5/5] Generated text\n{'='*60}")
    print(f"\nShared prefix: \"{shared_prompt}\"\n")
    for i, (cont, text) in enumerate(results):
        print(f"Agent {i+1}  (+\"{cont}\")")
        # Show the full generated output (trimmed to 150 chars)
        display = f"{shared_prompt}{text}"
        print(f"        → {display[:150]}{'…' if len(display)>150 else ''}")
        print()

    # ── Memory analysis ───────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("Memory Analysis")
    print(f"{'='*60}")

    bytes_per_tok = kv_bytes / prompt_len
    child_kv_bytes = sum(t * bytes_per_tok for t in child_new_tok_counts)

    naive_total    = n_agents * kv_bytes + child_kv_bytes   # each agent copies prompt KV
    agentkv_total  = kv_bytes           + child_kv_bytes   # prompt KV shared once

    savings_pct = (1 - agentkv_total / naive_total) * 100

    print(f"  Model           : {model_name}")
    print(f"  Shared prompt   : {prompt_len} tokens → {kv_bytes/1024:.1f} KB KV cache")
    print(f"  Agents          : {n_agents}")
    print(f"  Avg new tokens  : {sum(child_new_tok_counts)/n_agents:.0f}")
    print()
    print(f"  Naive KV memory : {naive_total/1024:.1f} KB")
    print(f"    ({n_agents}× prompt copy + per-agent new tokens)")
    print(f"  AgentKV memory  : {agentkv_total/1024:.1f} KB")
    print(f"    (1× shared prompt + per-agent new tokens)")
    print(f"  Saved           : {(naive_total-agentkv_total)/1024:.1f} KB  ({savings_pct:.1f}%)")
    print()
    print(f"  AgentKV stats   : {pool.stats()}")

    if device == "cuda":
        torch.cuda.synchronize()
        print(f"  Final VRAM used : {torch.cuda.memory_allocated()/1e9:.2f} GB")

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
        "naive_bytes": naive_total,
        "agentkv_bytes": agentkv_total,
        "savings_pct": savings_pct,
        "results": results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentKV + HuggingFace Model Demo")
    parser.add_argument("--model",  default="gpt2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="Scientists discovered that artificial intelligence systems")
    parser.add_argument("--n-agents",   type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=20)
    args = parser.parse_args()

    run_demo(
        model_name=args.model,
        device=args.device,
        shared_prompt=args.prompt,
        n_agents=args.n_agents,
        new_tokens_per_agent=args.new_tokens,
    )
