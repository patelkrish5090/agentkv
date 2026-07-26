"""
bench/hf_cache_demo.py — End-to-end HuggingFace demo with real GPU CoW.

Unlike `hf_model_demo.py` / `deepseek_stress_test.py` (which run AgentKV's
pool *alongside* a plain HF `DynamicCache` + `copy.deepcopy()`, and report
savings computed from a bytes-per-token formula), this script passes an
`AgentKVCache` (agentkv/hf_cache.py) directly to the model as
`past_key_values`. The model reads and writes real K/V tensors straight
into the AgentKVPool's slab storage, and forking an agent is a real
`AgentKVPool.fork()` ref-count bump — zero bytes copied unless two agents'
generation actually collides on a still-shared residual block. Every VRAM
number this script prints comes from `torch.cuda.memory_allocated()` /
`max_memory_allocated()`, not a formula.

Known-fragile point (real, not hypothetical — see tests/test_hf_cache.py)
--------------------------------------------------------------------------
`AgentKVPool.create_root()` stores the *entire* prompt's token ids on the
handle immediately, before any KV has actually been computed for it. But
`AgentKVCache.__init__` seeds `self._seen_tokens = len(handle.tokens)` —
so a freshly constructed cache over a root handle reports the full prompt
length as "already cached" before the first prefill forward has even run.
Left alone, the model's very first forward pass would compute cache
positions as if the prompt were being appended after a phantom cache of
the same length, producing wrong position embeddings and a wrong causal
mask. `tests/test_hf_cache.py` works around this by manually resetting
`cache._seen_tokens = 0` right after construction, before the first
`update()` call — this script does the same (see `root_cache._seen_tokens
= 0` below). Forked child caches do NOT need this reset — their handle's
token count matches KV data that was genuinely already written by the
parent.

Status: this is the first time AgentKVCache has been run through a real
model's generate() loop end-to-end (previously validated only against
synthetic tensors in tests/test_hf_cache.py, and against toy models like
gpt2 without quantization or a real memory audit). If you hit a shape or
position-id error on a model this hasn't been tried against, the seed-reset
above, or a transformers Cache-protocol method AgentKVCache doesn't
override, are the most likely causes — please share the traceback.

Usage
-----
# Fast smoke test (no quantization, tiny model):
python bench/hf_cache_demo.py --model gpt2 --device cuda

# Real model, real measured savings, T4-sized (16 GB):
pip install bitsandbytes
python bench/hf_cache_demo.py \\
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --load-in-4bit --device cuda --n-agents 4 --new-tokens 60
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentkv import AgentKVPool, PoolConfig
from agentkv.hf_cache import AgentKVCache

LONG_SYSTEM_PROMPT = (
    "You are an expert AI research assistant specialising in medical diagnostics, "
    "drug discovery, and clinical trial design. "
    "You have access to the latest literature from PubMed, ClinicalTrials.gov, "
    "and FDA drug approval databases. "
    "When answering questions, you always cite your sources, "
    "acknowledge uncertainty, and flag potential drug interactions or contraindications."
)

AGENT_QUERIES = [
    "What are the latest Phase III trials for GLP-1 receptor agonists in NASH?",
    "Summarise the mechanism of action of KRAS G12C inhibitors and their resistance pathways.",
    "Compare pembrolizumab and nivolumab efficacy in NSCLC with PD-L1 >=50%.",
    "What are the contraindications for SGLT2 inhibitors in CKD patients?",
    "Describe the pharmacokinetics of mRNA-LNP vaccines and their cold chain requirements.",
    "What is the current standard of care for HER2-positive metastatic breast cancer?",
]


def format_query(tokenizer, system_prompt: str, user_query: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_query} [/INST]"


def gpu_mem_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="4-bit NF4 quantization — required for 7B/8B models on a 16 GB T4")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--vram-fraction", type=float, default=0.3,
                         help="Fraction of *free* VRAM (after loading the model) given to the AgentKV pool")
    parser.add_argument("--prompt", type=str, default=LONG_SYSTEM_PROMPT)
    return parser.parse_args()


def main():
    args = parse_args()
    on_gpu = args.device == "cuda"

    print("=" * 65)
    print("AgentKV Phase 3a - Real GPU CoW Demo (measured, not computed)")
    print(f"  model      = {args.model}")
    print(f"  device     = {args.device}")
    print(f"  quantized  = {'4-bit NF4' if args.load_in_4bit else 'fp16'}")
    print(f"  agents     = {args.n_agents}")
    print(f"  new_tokens = {args.new_tokens}")
    print("=" * 65)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # AgentKVCache's slab pool is always float16 (see PoolConfig below), so the
    # model's own compute dtype must also be float16 everywhere - not just the
    # bnb_4bit_compute_dtype (which only governs the 4-bit dequant matmul).
    # Without this, transformers loads the model at its checkpoint's native
    # dtype (often bfloat16 for Llama-family models), and the query tensor
    # computed by the model's own attention layer ends up bfloat16 while
    # AgentKVCache hands back float16 key/value tensors - scaled_dot_product_
    # attention then rejects the dtype mismatch.
    compute_dtype = torch.float16 if on_gpu else torch.float32
    load_kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True, "dtype": compute_dtype}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not args.load_in_4bit:
        model = model.to(args.device)
    model.eval()

    mem_after_model = gpu_mem_gb() if on_gpu else 0.0
    print(f"   VRAM after model load: {mem_after_model:.2f} GB")

    mcfg = model.config
    num_layers = mcfg.num_hidden_layers
    num_kv_heads = getattr(mcfg, "num_key_value_heads",
                            getattr(mcfg, "num_attention_heads", 8))
    head_dim = getattr(mcfg, "head_dim", mcfg.hidden_size // mcfg.num_attention_heads)
    print(f"   {num_layers} layers | {num_kv_heads} KV heads | head_dim={head_dim}")

    # ── 2. AgentKV pool, sized from currently-free VRAM ───────────────────────
    print("\n[2/6] Initializing AgentKV pool...")
    if on_gpu:
        pool_cfg = PoolConfig.max_for_device(
            fraction=args.vram_fraction,
            num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
            block_size=16, dtype="float16", device="cuda",
        )
    else:
        pool_cfg = PoolConfig(
            total_blocks=256, block_size=16,
            num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
            dtype="float32", device=args.device,
        )
    pool = AgentKVPool(config=pool_cfg)
    mem_after_pool = gpu_mem_gb() if on_gpu else 0.0
    print(f"   {pool_cfg}")
    print(f"   VRAM after pool init: {mem_after_pool:.2f} GB "
          f"(+{mem_after_pool - mem_after_model:.2f} GB - real pool allocation, not a formula)")

    # ── 3. Real prefill through AgentKVCache ──────────────────────────────────
    print("\n[3/6] Prefilling shared prompt through AgentKVCache...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    prompt_ids = inputs.input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    root_handle = pool.create_root(prompt_ids)
    root_cache = AgentKVCache(pool, root_handle)
    root_cache._seen_tokens = 0  # see module docstring: required before the first update()

    t0 = time.perf_counter()
    with torch.inference_mode():
        model(**inputs, past_key_values=root_cache, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000

    safe_share_len = (prompt_len // pool.block_size) * pool.block_size
    if safe_share_len > 0:
        pool.commit_prefix(root_handle, safe_share_len)

    mem_after_prefill = gpu_mem_gb() if on_gpu else 0.0
    print(f"   {prompt_len} tokens | prefill: {prefill_ms:.0f} ms")
    print(f"   VRAM after prefill: {mem_after_prefill:.2f} GB")

    # ── 4. Fork agents — real zero-copy CoW, no copy.deepcopy() anywhere ──────
    print(f"\n[4/6] Forking {args.n_agents} agents (real CoW, no deepcopy)...")
    agent_caches = [root_cache.fork() for _ in range(args.n_agents)]
    mem_after_fork = gpu_mem_gb() if on_gpu else 0.0
    print(f"   VRAM after fork: {mem_after_fork:.2f} GB "
          f"(+{mem_after_fork - mem_after_prefill:.3f} GB for {args.n_agents} agents - "
          f"should be ~0, since forking only bumps ref counts)")

    # ── 5. Generate per agent ──────────────────────────────────────────────────
    print(f"\n[5/6] Generating {args.new_tokens} tokens per agent...")
    queries = (AGENT_QUERIES * ((args.n_agents // len(AGENT_QUERIES)) + 1))[:args.n_agents]
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    for i, (cache, query) in enumerate(zip(agent_caches, queries)):
        formatted = format_query(tokenizer, args.prompt, query)
        q_ids = tokenizer.encode(formatted, return_tensors="pt",
                                  add_special_tokens=False).to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            gen_ids = model.generate(
                q_ids,
                past_key_values=cache,
                max_new_tokens=args.new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        gen_ms = (time.perf_counter() - t0) * 1000

        gen_text = tokenizer.decode(gen_ids[0][q_ids.shape[1]:], skip_special_tokens=True)
        print(f"   Agent {i+1} [{gen_ms:.0f} ms]: "
              f"\"{gen_text[:80].strip().replace(chr(10), ' ')}\"")

    mem_after_gen = gpu_mem_gb() if on_gpu else 0.0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0.0

    # ── 6. Real measured summary ───────────────────────────────────────────────
    print(f"\n[6/6] Real measured VRAM summary\n{'='*65}")
    print(f"  After model load : {mem_after_model:.2f} GB")
    print(f"  After pool init  : {mem_after_pool:.2f} GB")
    print(f"  After prefill    : {mem_after_prefill:.2f} GB")
    print(f"  After {args.n_agents} forks    : {mem_after_fork:.2f} GB  "
          f"(delta +{mem_after_fork - mem_after_prefill:.3f} GB)")
    print(f"  After generation : {mem_after_gen:.2f} GB")
    print(f"  Peak during gen  : {peak_mem:.2f} GB")
    print(f"\n  AgentKV pool stats: {pool.stats()}")

    print(f"\nDone - {args.n_agents} agents, all numbers above are measured "
          f"from torch.cuda.memory_allocated(), not computed from a formula.")


if __name__ == "__main__":
    main()
