"""
bench/hf_cache_demo.py — End-to-end HuggingFace demo with real GPU CoW,
plus a real naive baseline for a genuine measured percentage-savings number.

Two backends, same workload, same instrumentation, so their printed VRAM
stage tables are directly diffable:

  --backend agentkv (default)
      `AgentKVCache` (agentkv/hf_cache.py) is passed directly to the model
      as `past_key_values`. The model reads/writes real K/V tensors into
      AgentKVPool's slab storage; forking an agent is a real
      `AgentKVPool.fork()` ref-count bump — zero bytes copied unless two
      agents' generation collides on a still-shared residual block.

  --backend naive
      Plain HF `DynamicCache`, and forking an agent means an actual
      `copy.deepcopy()` of the whole cache — the thing every other
      framework's naive per-request KV allocation is doing. This is the
      real baseline the agentkv backend is compared against; it is NOT a
      formula, it's the same workload measured the same way.

Every VRAM number either backend prints comes from
`torch.cuda.memory_allocated()` / `max_memory_allocated()`.

Known-fragile point in the agentkv backend (real, not hypothetical — see
tests/test_hf_cache.py)
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
`update()` call — this script does the same. Forked child caches do NOT
need this reset — their handle's token count matches KV data that was
genuinely already written by the parent.

Also known: the model's overall compute `dtype` must match AgentKVCache's
slab pool dtype (always float16) exactly, or scaled_dot_product_attention
rejects a query/key dtype mismatch — see the `compute_dtype` handling below.
This does not apply to the naive backend, whose cache dtype always matches
the model's own, whatever that is.

Usage
-----
# Fast smoke test (no quantization, tiny model, small prompt):
python bench/hf_cache_demo.py --model gpt2 --device cuda --backend agentkv

# Real model, real measured savings, T4-sized (16 GB), short prompt:
pip install bitsandbytes
python bench/hf_cache_demo.py \\
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --load-in-4bit --device cuda --n-agents 4 --new-tokens 60 \\
    --backend agentkv

# Bigger, more dramatic savings number: long shared prefix + more agents.
# Run BOTH backends (as two separate processes — don't compare within one
# process; a fresh CUDA context per run avoids any doubt about carried-over
# allocator state) on the identical workload, then diff the two summaries:
python bench/hf_cache_demo.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --load-in-4bit --device cuda --n-agents 8 --new-tokens 60 \\
    --prompt-preset long --backend agentkv
python bench/hf_cache_demo.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --load-in-4bit --device cuda --n-agents 8 --new-tokens 60 \\
    --prompt-preset long --backend naive
"""

import argparse
import copy
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentkv import AgentKVPool, PoolConfig
from agentkv.hf_cache import AgentKVCache

SHORT_PROMPT = (
    "You are an expert AI research assistant specialising in medical diagnostics, "
    "drug discovery, and clinical trial design. "
    "You have access to the latest literature from PubMed, ClinicalTrials.gov, "
    "and FDA drug approval databases. "
    "When answering questions, you always cite your sources, "
    "acknowledge uncertainty, and flag potential drug interactions or contraindications."
)

# A long, genuinely-written (not repeated/padded) shared context — stands in
# for a realistic RAG-retrieved document or long system prompt, to push the
# shared-prefix KV size from megabytes into the range where CoW savings show
# up as a real multi-hundred-MB/low-GB number instead of a rounding error.
LONG_PROMPT = (
    "You are an expert clinical AI assistant embedded in a hospital's decision-support "
    "system. The following is your complete reference context for this session; treat "
    "it as authoritative background and cite specific sections when relevant.\n\n"
    "SECTION 1 - DIABETES MANAGEMENT. Type 2 diabetes mellitus management has shifted "
    "substantially over the past decade with the introduction of GLP-1 receptor agonists "
    "and SGLT2 inhibitors as preferred second-line agents ahead of sulfonylureas, "
    "particularly in patients with established cardiovascular disease or chronic kidney "
    "disease. Semaglutide and tirzepatide have demonstrated significant reductions in "
    "major adverse cardiovascular events across multiple large outcome trials, alongside "
    "clinically meaningful weight loss. SGLT2 inhibitors such as empagliflozin and "
    "dapagliflozin reduce hospitalization for heart failure and slow progression of "
    "diabetic nephropathy, independent of glycemic control. Metformin remains first-line "
    "therapy absent contraindications such as an eGFR below 30 mL/min/1.73m^2 or acute "
    "risk of lactic acidosis. Insulin therapy is reserved for patients with severe "
    "hyperglycemia at diagnosis, or for those who fail to reach glycemic targets on "
    "combination oral and injectable non-insulin therapy.\n\n"
    "SECTION 2 - ONCOLOGY IMMUNOTHERAPY. Immune checkpoint inhibitors targeting PD-1, "
    "PD-L1, and CTLA-4 have transformed treatment paradigms across non-small cell lung "
    "cancer, melanoma, renal cell carcinoma, and increasingly earlier-stage disease "
    "settings. Response to checkpoint blockade correlates imperfectly with PD-L1 tumor "
    "proportion score, tumor mutational burden, and microsatellite instability status; "
    "none of these biomarkers alone reliably predicts individual patient response. "
    "Immune-related adverse events can affect nearly any organ system, most commonly "
    "presenting as colitis, pneumonitis, thyroiditis, hepatitis, or dermatitis, and "
    "management follows a graded corticosteroid escalation protocol with early "
    "specialist involvement for grade 3 or higher toxicity. Combination checkpoint "
    "blockade regimens improve response rates at the cost of substantially higher "
    "toxicity burden and require careful patient selection.\n\n"
    "SECTION 3 - ANTIMICROBIAL STEWARDSHIP. Empiric antibiotic selection for suspected "
    "sepsis should be guided by local resistance patterns, site of presumed infection, "
    "and patient-specific risk factors for multidrug-resistant organisms, with "
    "de-escalation to narrow-spectrum therapy once culture and sensitivity data return, "
    "typically within 48 to 72 hours. Overuse of broad-spectrum carbapenems and "
    "vancomycin without clear indication accelerates resistance development and "
    "increases risk of Clostridioides difficile infection. Procalcitonin-guided "
    "algorithms can support earlier antibiotic discontinuation in appropriately "
    "selected patients with resolving clinical status, reducing unnecessary antibiotic "
    "exposure days without increasing mortality or relapse rates.\n\n"
    "SECTION 4 - CARDIOVASCULAR RISK REDUCTION. Statin therapy intensity should be "
    "guided by calculated atherosclerotic cardiovascular disease risk, with high-intensity "
    "statins preferred for secondary prevention and for primary prevention patients above "
    "established risk thresholds. Non-statin adjuncts including ezetimibe and PCSK9 "
    "inhibitors are indicated when LDL cholesterol targets are not met on maximally "
    "tolerated statin therapy, particularly in patients with established atherosclerotic "
    "disease or genetically confirmed familial hypercholesterolemia. Blood pressure "
    "targets have trended lower following outcome trials demonstrating cardiovascular "
    "benefit from more intensive control, though this must be balanced against "
    "orthostatic hypotension and fall risk in older or frail patients.\n\n"
    "SECTION 5 - RESPIRATORY AND VACCINE GUIDANCE. mRNA-lipid nanoparticle vaccine "
    "platforms require strict cold-chain maintenance from manufacture through "
    "administration, with stability windows varying by formulation and dictating "
    "distribution logistics in resource-limited settings. Chronic obstructive pulmonary "
    "disease management follows a stepwise approach beginning with long-acting "
    "bronchodilator monotherapy, escalating to dual bronchodilator therapy and, in "
    "patients with frequent exacerbations and elevated eosinophil counts, addition of "
    "inhaled corticosteroids. Pulmonary rehabilitation and smoking cessation support "
    "remain the interventions with the strongest evidence for reducing exacerbation "
    "frequency and improving quality of life across disease severity.\n\n"
    "When answering questions, cite the relevant section number, acknowledge "
    "uncertainty where evidence is mixed, and flag potential drug interactions or "
    "contraindications explicitly rather than omitting them."
)

AGENT_QUERIES = [
    "What are the latest Phase III trials for GLP-1 receptor agonists in NASH?",
    "Summarise the mechanism of action of KRAS G12C inhibitors and their resistance pathways.",
    "Compare pembrolizumab and nivolumab efficacy in NSCLC with PD-L1 >=50%.",
    "What are the contraindications for SGLT2 inhibitors in CKD patients?",
    "Describe the pharmacokinetics of mRNA-LNP vaccines and their cold chain requirements.",
    "What is the current standard of care for HER2-positive metastatic breast cancer?",
    "When should procalcitonin-guided antibiotic de-escalation be avoided?",
    "Compare PCSK9 inhibitors and ezetimibe as adjuncts to statin therapy.",
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


def gpu_mem_gb(on_gpu: bool) -> float:
    if not on_gpu:
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action="store_true",
                         help="4-bit NF4 quantization — required for 7B/8B models on a 16 GB T4")
    parser.add_argument("--backend", choices=["agentkv", "naive"], default="agentkv",
                         help="agentkv = real AgentKVCache CoW fork; naive = real copy.deepcopy() per agent")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--vram-fraction", type=float, default=0.3,
                         help="Fraction of *free* VRAM (after loading the model) given to the AgentKV pool")
    parser.add_argument("--prompt-preset", choices=["short", "long"], default="short",
                         help="short = ~70-token prompt; long = ~1500-2000 token shared context, "
                              "for a bigger absolute savings number")
    parser.add_argument("--prompt", type=str, default=None,
                         help="Override the prompt entirely (ignores --prompt-preset)")
    return parser.parse_args()


def load_model(args, on_gpu: bool):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    return tokenizer, model


def print_summary(label: str, stages: dict, extra: str = "") -> None:
    print(f"\n[{label}] Real measured VRAM summary\n{'='*65}")
    for k, v in stages.items():
        print(f"  {k:<26}: {v:.3f} GB")
    if extra:
        print(f"\n{extra}")


def run_agentkv(model, tokenizer, args, prompt: str, on_gpu: bool) -> dict:
    stages = {}
    stages["after model load"] = gpu_mem_gb(on_gpu)

    mcfg = model.config
    num_layers = mcfg.num_hidden_layers
    num_kv_heads = getattr(mcfg, "num_key_value_heads", getattr(mcfg, "num_attention_heads", 8))
    head_dim = getattr(mcfg, "head_dim", mcfg.hidden_size // mcfg.num_attention_heads)
    print(f"   {num_layers} layers | {num_kv_heads} KV heads | head_dim={head_dim}")

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
    stages["after cache/pool init"] = gpu_mem_gb(on_gpu)
    print(f"   {pool_cfg}")

    print("\n[3/6] Prefilling shared prompt through AgentKVCache...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = inputs.input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    root_handle = pool.create_root(prompt_ids)
    root_cache = AgentKVCache(pool, root_handle)
    root_cache._seen_tokens = 0  # see module docstring

    t0 = time.perf_counter()
    with torch.inference_mode():
        model(**inputs, past_key_values=root_cache, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000

    safe_share_len = (prompt_len // pool.block_size) * pool.block_size
    if safe_share_len > 0:
        pool.commit_prefix(root_handle, safe_share_len)
    stages["after prefill"] = gpu_mem_gb(on_gpu)
    print(f"   {prompt_len} tokens | prefill: {prefill_ms:.0f} ms")

    print(f"\n[4/6] Forking {args.n_agents} agents (real CoW, no deepcopy)...")
    agent_caches = [root_cache.fork() for _ in range(args.n_agents)]
    stages[f"after {args.n_agents} forks"] = gpu_mem_gb(on_gpu)

    print(f"\n[5/6] Generating {args.new_tokens} tokens per agent...")
    queries = (AGENT_QUERIES * ((args.n_agents // len(AGENT_QUERIES)) + 1))[:args.n_agents]
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    for i, (cache, query) in enumerate(zip(agent_caches, queries)):
        formatted = format_query(tokenizer, prompt, query)
        q_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen_ids = model.generate(
                q_ids, past_key_values=cache, max_new_tokens=args.new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        gen_ms = (time.perf_counter() - t0) * 1000
        gen_text = tokenizer.decode(gen_ids[0][q_ids.shape[1]:], skip_special_tokens=True)
        print(f"   Agent {i+1} [{gen_ms:.0f} ms]: \"{gen_text[:80].strip().replace(chr(10), ' ')}\"")

    stages["after generation"] = gpu_mem_gb(on_gpu)
    stages["peak during generation"] = torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0.0

    print_summary("6/6 agentkv", stages, extra=f"AgentKV pool stats: {pool.stats()}")
    return stages


def run_naive(model, tokenizer, args, prompt: str, on_gpu: bool) -> dict:
    stages = {}
    stages["after model load"] = gpu_mem_gb(on_gpu)
    stages["after cache/pool init"] = gpu_mem_gb(on_gpu)  # nothing to init for naive

    print("\n[3/6] Prefilling shared prompt through a plain HF DynamicCache...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = inputs.input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    t0 = time.perf_counter()
    with torch.inference_mode():
        prefix_out = model(**inputs, use_cache=True)
    prefill_ms = (time.perf_counter() - t0) * 1000
    shared_cache = to_dynamic_cache(prefix_out.past_key_values)
    stages["after prefill"] = gpu_mem_gb(on_gpu)
    print(f"   {prompt_len} tokens | prefill: {prefill_ms:.0f} ms")

    print(f"\n[4/6] Cloning cache for {args.n_agents} agents (real copy.deepcopy(), not CoW)...")
    agent_caches = [copy.deepcopy(shared_cache) for _ in range(args.n_agents)]
    stages[f"after {args.n_agents} forks"] = gpu_mem_gb(on_gpu)

    print(f"\n[5/6] Generating {args.new_tokens} tokens per agent...")
    queries = (AGENT_QUERIES * ((args.n_agents // len(AGENT_QUERIES)) + 1))[:args.n_agents]
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    for i, (cache, query) in enumerate(zip(agent_caches, queries)):
        formatted = format_query(tokenizer, prompt, query)
        q_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen_ids = model.generate(
                q_ids, past_key_values=cache, max_new_tokens=args.new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        gen_ms = (time.perf_counter() - t0) * 1000
        gen_text = tokenizer.decode(gen_ids[0][q_ids.shape[1]:], skip_special_tokens=True)
        print(f"   Agent {i+1} [{gen_ms:.0f} ms]: \"{gen_text[:80].strip().replace(chr(10), ' ')}\"")

    stages["after generation"] = gpu_mem_gb(on_gpu)
    stages["peak during generation"] = torch.cuda.max_memory_allocated() / 1e9 if on_gpu else 0.0

    print_summary("6/6 naive", stages)
    return stages


def main():
    args = parse_args()
    on_gpu = args.device == "cuda"
    prompt = args.prompt if args.prompt is not None else (
        LONG_PROMPT if args.prompt_preset == "long" else SHORT_PROMPT
    )

    print("=" * 65)
    print(f"AgentKV Phase 3a - {args.backend} backend (measured, not computed)")
    print(f"  model         = {args.model}")
    print(f"  device        = {args.device}")
    print(f"  quantized     = {'4-bit NF4' if args.load_in_4bit else 'fp16'}")
    print(f"  backend       = {args.backend}")
    print(f"  agents        = {args.n_agents}")
    print(f"  new_tokens    = {args.new_tokens}")
    print(f"  prompt_preset = {args.prompt_preset} ({len(prompt.split())} words)")
    print("=" * 65)

    print(f"\n[1/6] Loading {args.model}...")
    tokenizer, model = load_model(args, on_gpu)

    if args.backend == "agentkv":
        run_agentkv(model, tokenizer, args, prompt, on_gpu)
    else:
        run_naive(model, tokenizer, args, prompt, on_gpu)

    print(f"\nDone - backend={args.backend}, {args.n_agents} agents, all numbers above "
          f"are measured from torch.cuda.memory_allocated(), not computed from a formula.")


if __name__ == "__main__":
    main()
