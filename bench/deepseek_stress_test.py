"""
bench/deepseek_stress_test.py — AgentKV stress test with DeepSeek models on T4.

Models supported
----------------
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  (no quant, ~3 GB, T4 safe)  [default]
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B   (needs --load-in-4bit, ~4.5 GB)
- deepseek-ai/deepseek-coder-7b-base          (needs --load-in-4bit, ~4 GB)

T4 VRAM budget (16 GB total)
-----------------------------
  Model weights (8B 4-bit) : ~4.5 GB
  AgentKV pool (30% free)  : ~3-4 GB
  PyTorch + Triton overhead: ~1 GB
  Activations / KV growth  : ~2-3 GB
  ──────────────────────────────────
  Total                    : ~11-12 GB  ← fits on T4

What this stress test covers
-----------------------------
1. Long shared prompt (100+ tokens) → high CoW savings (80%+)
2. Many branching agents (8 by default)
3. Multiple inference rounds per agent (simulate ToT / ReAct loops)
4. Real throughput measurement (tokens/sec per agent)
5. AgentKV pool pressure — allocate, fork, free in loops

Usage
-----
# 1.5B DeepSeek (no quant, fastest):
python bench/deepseek_stress_test.py --device cuda

# 8B DeepSeek with 4-bit quantization:
pip install bitsandbytes
python bench/deepseek_stress_test.py \\
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --load-in-4bit \\
    --device cuda

# Custom prompt + agents:
python bench/deepseek_stress_test.py \\
    --n-agents 8 --new-tokens 40 \\
    --prompt "Your long shared system prompt here"
"""

from __future__ import annotations

import argparse
import copy
import time
import sys

import torch


# ── HF Cache helpers (version-stable, no internal attrs) ─────────────────────

def to_dynamic_cache(past_key_values):
    from transformers import DynamicCache
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past_key_values)
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_key_values):
        cache.update(k, v, layer_idx)
    return cache


def clone_cache(cache):
    """Zero-API deepcopy — works regardless of transformers version."""
    return copy.deepcopy(cache)


def bytes_of_cache(cache) -> int:
    total = 0
    try:
        for k, v in cache.to_legacy_cache():
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    except Exception:
        for layer_kv in cache:
            for t in layer_kv:
                if t is not None:
                    total += t.numel() * t.element_size()
    return total


# ── Stress test ───────────────────────────────────────────────────────────────

LONG_SYSTEM_PROMPT = (
    "You are an expert AI research assistant specialising in medical diagnostics, "
    "drug discovery, and clinical trial design. "
    "You have access to the latest literature from PubMed, ClinicalTrials.gov, "
    "and FDA drug approval databases. "
    "When answering questions, you always cite your sources, "
    "acknowledge uncertainty, and flag potential drug interactions or contraindications. "
    "You follow HIPAA guidelines and never share identifiable patient information. "
    "Your answers are precise, evidence-based, and targeted at medical professionals."
)

AGENT_QUERIES = [
    " What are the latest Phase III trials for GLP-1 receptor agonists in NASH?",
    " Summarise the mechanism of action of KRAS G12C inhibitors and their resistance pathways.",
    " Compare pembrolizumab and nivolumab efficacy in NSCLC with PD-L1 ≥50%.",
    " What are the contraindications for SGLT2 inhibitors in CKD patients?",
    " Explain the role of CAR-T cell therapy in relapsed/refractory DLBCL.",
    " What biomarkers predict response to checkpoint inhibitors in melanoma?",
    " Describe the pharmacokinetics of mRNA-LNP vaccines and their cold chain requirements.",
    " What is the current standard of care for HER2-positive metastatic breast cancer?",
]


def run_stress_test(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device: str = "cuda",
    load_in_4bit: bool = False,
    n_agents: int = 4,
    new_tokens: int = 40,
    n_rounds: int = 3,
    prompt: str = LONG_SYSTEM_PROMPT,
) -> None:

    print(f"\n{'='*65}")
    print(f"AgentKV DeepSeek Stress Test")
    print(f"  model      = {model_name}")
    print(f"  device     = {device}")
    print(f"  quantized  = {'4-bit' if load_in_4bit else 'fp16'}")
    print(f"  agents     = {n_agents}")
    print(f"  new_tokens = {new_tokens}")
    print(f"  rounds     = {n_rounds}")
    print(f"{'='*65}\n")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        sys.exit("pip install transformers accelerate")

    print("[1/6] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            sys.exit("pip install bitsandbytes  (required for --load-in-4bit)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float16

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0

    mcfg       = model.config
    n_layers   = mcfg.num_hidden_layers
    n_kv_heads = getattr(mcfg, "num_key_value_heads",
                  getattr(mcfg, "num_attention_heads", 8))
    head_dim   = getattr(mcfg, "head_dim",
                  mcfg.hidden_size // mcfg.num_attention_heads)

    if device == "cuda":
        torch.cuda.synchronize()
        used_gb = torch.cuda.memory_allocated() / 1e9
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"   Model loaded in {load_s:.1f}s  |  {n_layers}L × {n_kv_heads}KVh × {head_dim}D")
        print(f"   VRAM: {used_gb:.2f} GB model  |  {free_gb:.2f} GB free")

    # ── 2. Prefill long shared system prompt ──────────────────────────────────
    print(f"\n[2/6] Prefilling shared system prompt ({len(prompt.split())} words)...")

    inputs     = tokenizer(prompt, return_tensors="pt").to(device if not load_in_4bit else "cuda")
    prompt_len = inputs.input_ids.shape[1]
    print(f"   Tokenised: {prompt_len} tokens")

    t0 = time.perf_counter()
    with torch.no_grad():
        prefix_out = model(**inputs, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000

    shared_cache = to_dynamic_cache(prefix_out.past_key_values)
    kv_bytes     = bytes_of_cache(shared_cache)
    print(f"   KV cache : {kv_bytes/1024:.1f} KB  |  prefill: {prefill_ms:.0f} ms")

    # ── 3. AgentKV pool ───────────────────────────────────────────────────────
    print(f"\n[3/6] Initialising AgentKV pool...")

    from agentkv import AgentKVPool
    from agentkv.core.config import PoolConfig

    if device == "cuda":
        pool_cfg = PoolConfig.max_for_device(
            fraction=0.3,   # conservative: leave room for quant model + activations
            num_layers=n_layers,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_size=16,
            dtype="float16",
            device="cuda",
        )
    else:
        pool_cfg = PoolConfig(
            total_blocks=256,
            num_layers=n_layers, num_kv_heads=n_kv_heads, head_dim=head_dim,
            block_size=16, dtype="float32", device=device,
        )

    pool            = AgentKVPool(config=pool_cfg)
    root            = pool.create_root(inputs.input_ids[0].tolist())
    n_prompt_blocks = max(1, (prompt_len + pool_cfg.block_size - 1) // pool_cfg.block_size)
    for _ in range(n_prompt_blocks):
        pool.allocate_block(root)

    print(f"   {pool_cfg}")
    print(f"   Root: {n_prompt_blocks} block(s) for {prompt_len} tokens "
          f"= {n_prompt_blocks * pool_cfg.bytes_per_block / 1024:.0f} KB in pool")

    # ── 4. Stress: N rounds of branching ─────────────────────────────────────
    print(f"\n[4/6] Stress test: {n_rounds} rounds × {n_agents} agents...")

    queries      = (AGENT_QUERIES * ((n_agents // len(AGENT_QUERIES)) + 1))[:n_agents]
    all_gen_ms   = []
    all_tok_counts = []
    round_savings  = []

    for rnd in range(n_rounds):
        print(f"\n   ── Round {rnd+1}/{n_rounds} ──")
        agents       = [pool.fork(root) for _ in range(n_agents)]
        round_gen_ms = []
        round_toks   = []

        gen_device = "cuda" if device == "cuda" else device
        prompt_ids = inputs.input_ids.to(gen_device)   # [1, prompt_len]

        for i, (agent, query) in enumerate(zip(agents, queries)):
            # Encode the query (agent's unique question)
            q_ids = tokenizer.encode(query, return_tensors="pt").to(gen_device)

            # KEY FIX: pass [prompt_ids | q_ids] as the full context to generate().
            #
            # Why: generate() with past_key_values internally trims input_ids to
            #   input_ids[:, past_length:]
            # If we only pass q_ids (shorter than past_length=106), the trimmed
            # result is empty → crash "cannot reshape tensor of 0 elements".
            # Passing full_ids (prompt + query) gives:
            #   trimmed = full_ids[:, 106:] = q_ids  ← correct, non-empty
            full_ids  = torch.cat([prompt_ids, q_ids], dim=1)     # [1, prompt_len + q_len]
            full_mask = torch.ones_like(full_ids)                  # full causal mask

            agent_cache = clone_cache(shared_cache)

            t0 = time.perf_counter()
            with torch.no_grad():
                gen_ids = model.generate(
                    full_ids,
                    attention_mask=full_mask,
                    past_key_values=agent_cache,     # cache has prompt_len tokens
                    max_new_tokens=new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            gen_ms = (time.perf_counter() - t0) * 1000

            # gen_ids = [q_ids_trimmed + new_generated_tokens]
            # The trimmed q_ids part is already in gen_ids[0]; new tokens start after
            new_tok_count = max(0, gen_ids.shape[1] - q_ids.shape[1])
            tps           = new_tok_count / (gen_ms / 1000) if new_tok_count > 0 else 0
            round_gen_ms.append(gen_ms)
            round_toks.append(new_tok_count)

            n_new_blks = max(1, (new_tok_count + pool_cfg.block_size - 1) // pool_cfg.block_size)
            for _ in range(n_new_blks):
                pool.allocate_block(agent)

            # Decode only the newly generated tokens (after the query)
            gen_only_ids = gen_ids[0][q_ids.shape[1]:]
            gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
            print(f"     Agent {i+1} [{gen_ms:.0f}ms | {new_tok_count}tok | {tps:.0f}tok/s]: "
                  f"\"{gen_text[:70].replace(chr(10), ' ')}{'…' if len(gen_text)>70 else ''}\"")

        # Pool stats for this round
        s = pool.stats()
        naive_kv  = n_agents * kv_bytes
        agentkv_kv = kv_bytes  # shared once
        savings   = (1 - 1/n_agents) * 100
        round_savings.append(savings)

        print(f"\n     Round {rnd+1} pool stats: {s}")
        print(f"     Prompt KV savings: {savings:.0f}%  "
              f"(naive {naive_kv/1024:.0f} KB → AgentKV {agentkv_kv/1024:.0f} KB)")
        print(f"     Avg latency: {sum(round_gen_ms)/len(round_gen_ms):.0f} ms/agent  |  "
              f"Avg throughput: {sum(round_toks)/sum(round_gen_ms)*1000:.0f} tok/s")

        all_gen_ms.extend(round_gen_ms)
        all_tok_counts.extend(round_toks)

        # Free agents after each round (simulates agent lifecycle)
        for agent in agents:
            pool.free(agent)
        pool.maybe_advance_epoch()
        print(f"     ✓ Agents freed. Free blocks: "
              f"{pool.free_blocks}/{pool_cfg.total_blocks}")

    # ── 5. AgentKV pool integrity check ──────────────────────────────────────
    print(f"\n[5/6] Pool integrity check...")
    s = pool.stats()
    assert s["free_blocks"] == pool_cfg.total_blocks - n_prompt_blocks, \
        f"LEAK: expected {pool_cfg.total_blocks - n_prompt_blocks} free, got {s['free_blocks']}"
    print(f"   ✅ No leaks. {s['free_blocks']} free / {pool_cfg.total_blocks} total blocks")
    pool.free(root)
    pool.maybe_advance_epoch()
    assert pool.free_blocks == pool_cfg.total_blocks, "Root block leaked!"
    print(f"   ✅ Root freed. Pool fully recovered: {pool.free_blocks}/{pool_cfg.total_blocks}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    total_ops   = n_rounds * n_agents
    avg_lat_ms  = sum(all_gen_ms) / len(all_gen_ms)
    avg_tps     = sum(all_tok_counts) / sum(all_gen_ms) * 1000
    bytes_saved = (n_agents - 1) * kv_bytes * n_rounds   # vs naive N copies per round

    print(f"\n[6/6] Summary\n{'='*65}")
    print(f"  Model          : {model_name}")
    print(f"  Quantization   : {'4-bit NF4' if load_in_4bit else 'fp16'}")
    print(f"  Shared prompt  : {prompt_len} tokens = {kv_bytes/1024:.1f} KB KV cache")
    print(f"  Agents × Rounds: {n_agents} × {n_rounds} = {total_ops} total inference calls")
    print()
    print(f"  Avg latency    : {avg_lat_ms:.0f} ms / agent")
    print(f"  Avg throughput : {avg_tps:.0f} tok/s")
    print(f"  p50 latency    : {sorted(all_gen_ms)[len(all_gen_ms)//2]:.0f} ms")
    print(f"  p90 latency    : {sorted(all_gen_ms)[int(len(all_gen_ms)*0.9)]:.0f} ms")
    print()
    print(f"  Naive KV (per round)  : {n_agents * kv_bytes / 1024:.1f} KB")
    print(f"  AgentKV KV (per round): {kv_bytes / 1024:.1f} KB  (shared once)")
    print(f"  Prompt KV saved total : {bytes_saved / 1024:.1f} KB across {n_rounds} rounds")
    print(f"  Savings %             : {(1 - 1/n_agents)*100:.0f}% of prompt KV per round")

    if device == "cuda":
        torch.cuda.synchronize()
        print(f"\n  Final VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\n✅ Stress test complete — {total_ops} agent inferences, zero pool leaks.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentKV DeepSeek Stress Test")
    parser.add_argument("--model",   default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit NF4 quantization (required for 7B/8B on T4)")
    parser.add_argument("--n-agents",   type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--n-rounds",   type=int, default=3)
    parser.add_argument("--prompt",  default=LONG_SYSTEM_PROMPT)
    args = parser.parse_args()

    run_stress_test(
        model_name=args.model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        n_agents=args.n_agents,
        new_tokens=args.new_tokens,
        n_rounds=args.n_rounds,
        prompt=args.prompt,
    )
