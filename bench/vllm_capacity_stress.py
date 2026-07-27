"""
bench/vllm_capacity_stress.py — real vLLM block-pool capacity comparison.

bench/vllm_real_user_sim.py measures torch.cuda.memory_allocated(), which
turned out to be the wrong lens for a whole-engine comparison: vLLM's
CacheEngine pre-reserves a FIXED block pool sized by gpu_memory_utilization
regardless of which BlockSpaceManager governs it - swapping AgentKV in
can't change that number at all. The real payoff of CoW sharing is "how
many more concurrent agents fit in that SAME fixed pool," not a smaller
memory_allocated() reading.

This script measures that directly: it polls the scheduler's own
block_manager.get_num_free_gpu_blocks() from a background thread while
generate() runs (both backends implement this identically - it's part of
the abstract BlockSpaceManager interface), tracking the minimum free-block
count actually observed - i.e. real peak concurrent block usage, read from
vLLM's own live state, not inferred.

Usage
-----
# Deliberately small gpu_memory_utilization to make capacity pressure
# reachable without a huge model or absurd sequence counts.
python bench/vllm_capacity_stress.py --backend agentkv \\
    --sweep 4,8,16,32,64 --gpu-memory-utilization 0.1
python bench/vllm_capacity_stress.py --backend stock \\
    --sweep 4,8,16,32,64 --gpu-memory-utilization 0.1
"""

import argparse
import os
import sys
import threading
import time

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SHARED_CONTEXT = (
    "You are a customer support AI assistant for a cloud infrastructure "
    "company. You have access to the following internal knowledge base "
    "context for this session:\n\n"
    "INCIDENT HISTORY: On 2026-06-14, a regional network partition caused "
    "elevated API latency for approximately 40 minutes across the us-east-2 "
    "region, affecting object storage and queue services; root cause was a "
    "misconfigured BGP route announced during a routine maintenance window. "
    "On 2026-07-02, a database connection pool exhaustion issue caused "
    "intermittent 503 errors for customers on the legacy API gateway; "
    "mitigated by an emergency connection limit increase and later resolved "
    "by a permanent pool-sizing fix.\n\n"
    "BILLING POLICY: Customers are billed monthly in arrears based on "
    "metered usage across compute, storage, and network egress. Service "
    "credits for documented outages are calculated as a percentage of that "
    "month's affected-service charges, prorated to incident duration, and "
    "must be requested within 30 days of the incident.\n\n"
    "Answer customer questions using only the above context where relevant, "
    "citing the incident date if referencing an outage."
)

USER_QUESTIONS = [
    "Was my account affected by any outage in the last two months?",
    "How is my service credit calculated for the June incident?",
    "Why did I see 503 errors on the API gateway in early July?",
    "What's the difference between Tier 1 and Tier 2 support?",
    "Can you explain the root cause of the June 14th latency issue?",
    "Do I need to request a credit manually or is it automatic?",
    "Is the legacy API gateway issue fully resolved now?",
    "What counts as a documented outage for credit purposes?",
]


def build_prompts(n_users: int) -> list:
    questions = (USER_QUESTIONS * ((n_users // len(USER_QUESTIONS)) + 1))[:n_users]
    return [f"{SHARED_CONTEXT}\n\nCustomer question: {q}\nAnswer:" for q in questions]


class FreeBlockPoller:
    """Polls the scheduler's live free-block count from a background thread
    while llm.generate() runs in the main thread (vLLM's engine runs
    in-process and synchronously, unlike SGLang's subprocess architecture,
    so this is a direct read of real state, not an external proxy)."""

    def __init__(self, block_manager, interval_s: float = 0.02):
        self.block_manager = block_manager
        self.interval_s = interval_s
        self.min_free = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            free = self.block_manager.get_num_free_gpu_blocks()
            self.min_free = free if self.min_free is None else min(self.min_free, free)
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval_s)

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sample()  # one last read after generate() returns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--backend", choices=["agentkv", "stock"], default="agentkv")
    parser.add_argument("--sweep", type=str, default="4,8,16,32",
                         help="Comma-separated list of n_users values to test, "
                              "each combined with --n-agents-per-user completions")
    parser.add_argument("--n-agents-per-user", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.1,
                         help="Deliberately small - makes capacity pressure reachable. "
                              "Raise this (e.g. 0.7) for a large model like an 8B checkpoint, "
                              "where weights alone need real headroom before any KV blocks fit.")
    parser.add_argument("--quantization", type=str, default=None,
                         help="'bitsandbytes' for real in-flight 4-bit NF4 quantization of a plain "
                              "fp16/bf16 checkpoint (confirmed present in vLLM v0.5.4). Loading with "
                              "this on an 8B model genuinely takes ~10+ minutes with no progress "
                              "output during the BitsAndBytes weight-loading step - that's normal, "
                              "not a hang; vLLM's own log says this path 'is not fully optimized "
                              "yet'. 'awq'/'gptq' are faster alternatives if you have a pre-quantized "
                              "checkpoint already.")
    parser.add_argument("--max-model-len", type=int, default=None,
                         help="Cap the engine's max sequence length - some checkpoints (e.g. "
                              "DeepSeek-R1-Distill-Llama-8B, inherited from Llama 3.1's 128K context) "
                              "default to a max_model_len vLLM then insists the reserved KV cache must "
                              "be able to hold in full for at least one sequence, causing a startup "
                              "ValueError ('max seq len is larger than the maximum number of tokens "
                              "that can be stored in KV cache') at low --gpu-memory-utilization even "
                              "though this benchmark never sends prompts anywhere near that long.")
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sweep = [int(x) for x in args.sweep.split(",")]

    print("=" * 65)
    print(f"AgentKV vLLM capacity stress - backend={args.backend}")
    print(f"  model                   = {args.model}")
    print(f"  gpu_memory_utilization  = {args.gpu_memory_utilization}")
    print(f"  sweep (n_users)         = {sweep}")
    print(f"  n_agents_per_user       = {args.n_agents_per_user}")
    print("=" * 65)

    import vllm
    from vllm import SamplingParams

    if args.backend == "agentkv":
        from integrations.vllm import patch_vllm
        patch_vllm()
    else:
        print("\n[stock] Running with vLLM's own unmodified BlockSpaceManagerV2.")

    print(f"\nLoading {args.model}...")
    llm_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        # T4 is Turing - no native bf16 tensor cores. Pin float16 rather
        # than trust "auto" dtype detection (often bfloat16 for Llama-
        # family checkpoints' config.json).
        dtype="float16",
        use_v2_block_manager=True,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
        if args.quantization == "bitsandbytes":
            llm_kwargs["load_format"] = "bitsandbytes"
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = vllm.LLM(**llm_kwargs)

    block_manager = llm.llm_engine.scheduler[0].block_manager
    total_blocks = llm.llm_engine.cache_config.num_gpu_blocks
    print(f"Total GPU KV blocks available (fixed, same for both backends): {total_blocks}")

    print(f"\n{'n_users':>8} {'total_seqs':>11} {'peak_blocks_used':>17} "
          f"{'peak_pct':>9} {'wall_s':>8} {'status':>10}")

    results = []
    for n_users in sweep:
        prompts = build_prompts(n_users)
        sampling_params = SamplingParams(
            n=args.n_agents_per_user, max_tokens=args.max_tokens,
            temperature=0.8, top_p=0.95,
        )
        status = "ok"
        t0 = time.perf_counter()
        try:
            with FreeBlockPoller(block_manager) as poller:
                llm.generate(prompts, sampling_params, use_tqdm=False)
            wall_s = time.perf_counter() - t0
            peak_used = total_blocks - poller.min_free if poller.min_free is not None else None
        except Exception as e:
            wall_s = time.perf_counter() - t0
            peak_used = None
            status = f"FAILED: {type(e).__name__}"

        total_seqs = n_users * args.n_agents_per_user
        peak_pct = (peak_used / total_blocks * 100) if peak_used is not None else float("nan")
        print(f"{n_users:>8} {total_seqs:>11} {str(peak_used):>17} "
              f"{peak_pct:>8.1f}% {wall_s:>7.2f}s {status:>10}")
        results.append(dict(
            backend=args.backend, n_users=n_users, total_seqs=total_seqs,
            total_blocks=total_blocks, peak_blocks_used=peak_used,
            peak_pct=peak_pct, wall_s=wall_s, status=status,
        ))

        if status != "ok":
            print(f"\nStopped sweep early at n_users={n_users} (backend ran out of "
                  f"capacity or errored) - this IS the interesting data point.")
            break

    print(f"\nDone - backend={args.backend}. Compare peak_blocks_used at each "
          f"n_users against the same sweep with the other --backend value.")


if __name__ == "__main__":
    main()
