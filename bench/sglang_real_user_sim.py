"""
bench/sglang_real_user_sim.py — Real SGLang engine, real users, real GPU.

Higher risk than bench/vllm_real_user_sim.py — read this whole docstring
before running.

Known risks (real, checked before writing this - not hypothetical)
--------------------------------------------------------------------------
1. `register_radix_cache_backend` / `ServerArgs.radix_cache_backend` (the
   plugin point integrations/sglang/radix_cache.py registers against) could
   not be found in any released SGLang documentation, blog post, or cached
   search index as of 2026-07-26 - it appears to be a very recent/unreleased
   addition on SGLang's `main` branch. It may not exist yet in whatever
   `pip install sglang` actually gives you. This script checks for it
   up front (see `_check_registry_available` below) and exits with a clear
   message instead of a confusing crash if it's missing - if you hit that
   message, it means the integration's target API isn't in your installed
   version yet, not that AgentKV is broken.
2. SGLang's `Engine` spawns separate tokenizer/scheduler/detokenizer
   *subprocesses* (ZMQ-based) that actually load the model and run CUDA.
   `torch.cuda.memory_allocated()` in *this* process would read zero or
   garbage - it's not the process actually holding the GPU memory. This
   script polls `nvidia-smi` from a background thread instead, which is
   the only accurate way to measure this from outside SGLang's own process
   tree.
3. SGLang requires Python >= 3.10 and defaults to CUDA 13 wheels; Colab's
   preinstalled CUDA version may not match, requiring the cu12x wheel
   variant from docs.sglang.ai/whl/ instead of a plain `pip install sglang`.

Usage
-----
pip install sglang
python bench/sglang_real_user_sim.py --model facebook/opt-125m \\
    --backend agentkv --n-users 2 --n-agents-per-user 2 --max-tokens 20
"""

import argparse
import subprocess
import sys
import threading
import time


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
]


def build_prompts(n_users: int) -> list:
    questions = (USER_QUESTIONS * ((n_users // len(USER_QUESTIONS)) + 1))[:n_users]
    return [f"{SHARED_CONTEXT}\n\nCustomer question: {q}\nAnswer:" for q in questions]


def _check_registry_available() -> bool:
    try:
        from sglang.srt.mem_cache.registry import register_radix_cache_backend  # noqa: F401
        return True
    except ImportError:
        return False


class GpuMemPoller:
    """Polls nvidia-smi in a background thread - SGLang's Engine runs the
    model in a separate process tree, so torch.cuda.* in this process
    doesn't see that memory at all."""

    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return max(int(x.strip()) for x in out.stdout.strip().splitlines())

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, self._sample())
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        self.peak_mb = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--backend", choices=["agentkv", "stock"], default="agentkv")
    parser.add_argument("--n-users", type=int, default=4)
    parser.add_argument("--n-agents-per-user", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(f"AgentKV real-SGLang-engine simulation - backend={args.backend}")
    print(f"  model              = {args.model}")
    print(f"  n_users            = {args.n_users}")
    print(f"  n_agents_per_user  = {args.n_agents_per_user}")
    print(f"  max_tokens         = {args.max_tokens}")
    print("=" * 65)

    if args.backend == "agentkv":
        if not _check_registry_available():
            sys.exit(
                "\nSTOP: sglang.srt.mem_cache.registry.register_radix_cache_backend "
                "is not importable in this SGLang install.\n"
                "This means the plugin point integrations/sglang/radix_cache.py "
                "targets isn't present in whatever version `pip install sglang` "
                "gave you (it was researched off SGLang's main branch and may be "
                "unreleased). This is expected-possible, not a crash to debug - "
                "see the module docstring's risk #1. Run with --backend stock to "
                "at least confirm plain SGLang works in this environment, and "
                "report the `pip show sglang` version back so the integration "
                "can be checked against it."
            )
        from integrations.sglang import register
        register()
        radix_cache_backend = "agentkv"
    else:
        radix_cache_backend = None

    import sglang as sgl

    print(f"\n[1/3] Loading {args.model} into SGLang Engine...")
    t0 = time.perf_counter()
    engine_kwargs = dict(model_path=args.model, mem_fraction_static=args.mem_fraction_static)
    if radix_cache_backend:
        engine_kwargs["radix_cache_backend"] = radix_cache_backend
    engine = sgl.Engine(**engine_kwargs)
    load_s = time.perf_counter() - t0
    print(f"   Loaded in {load_s:.1f}s")

    prompts = build_prompts(args.n_users)
    sampling_params = [
        {"n": args.n_agents_per_user, "max_new_tokens": args.max_tokens,
         "temperature": 0.0 if args.n_agents_per_user == 1 else 0.8}
        for _ in prompts
    ]

    print(f"\n[2/3] Generating for {args.n_users} concurrent users "
          f"x {args.n_agents_per_user} completions each...")
    with GpuMemPoller() as poller:
        t0 = time.perf_counter()
        outputs = engine.generate(prompt=prompts, sampling_params=sampling_params)
        gen_s = time.perf_counter() - t0
    peak_mb = poller.peak_mb

    total_completions = 0
    total_new_tokens = 0
    for req_idx, out in enumerate(outputs):
        completions = out if isinstance(out, list) else [out]
        for comp_idx, completion in enumerate(completions):
            total_completions += 1
            meta = completion.get("meta_info", {}) if isinstance(completion, dict) else {}
            total_new_tokens += meta.get("completion_tokens", args.max_tokens)
            if req_idx == 0 and comp_idx == 0:
                text = completion.get("text", "") if isinstance(completion, dict) else str(completion)
                print(f"   [sample output] user 1, completion 1: \"{text[:80].strip()}\"")

    throughput = total_new_tokens / gen_s if gen_s > 0 else 0.0

    print(f"\n[3/3] Real measured results\n{'='*65}")
    print(f"  Backend                 : {args.backend}")
    print(f"  Total sequences         : {total_completions} "
          f"({args.n_users} users x {args.n_agents_per_user} completions)")
    print(f"  Total new tokens (est.) : {total_new_tokens}")
    print(f"  Generation wall time    : {gen_s:.2f} s")
    print(f"  Throughput              : {throughput:.1f} tokens/sec")
    print(f"  Peak whole-GPU memory   : {peak_mb} MB  (nvidia-smi polled - see risk #2)")
    print(f"\nDone - backend={args.backend}.")


if __name__ == "__main__":
    main()
