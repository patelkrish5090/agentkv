"""
bench/vllm_real_user_sim.py — Real vLLM engine, real users, real GPU.

Everything before this script exercised AgentKV's own APIs directly
(AgentKVPool, AgentKVCache) or the vLLM integration against a local stub of
vLLM's interfaces (tests/fakes/vllm/), because vLLM cannot be installed on
this project's Windows dev machine (needs Triton + a CUDA build). On a real
CUDA Linux box (e.g. Colab), vLLM CAN be installed for real — this script is
the first attempt at running AgentKVBlockManager inside vLLM's actual
LLMEngine and Scheduler, not a stand-in for them.

Simulated workload ("some real user using this")
--------------------------------------------------
A support/RAG-style deployment: one long shared system context (a realistic
stand-in for a retrieved document or long system prompt), several distinct
concurrent user questions submitted together (--n-users), and each user's
turn explored with N parallel completions (--n-agents-per-user, via vLLM's
`SamplingParams(n=...)` — this is what actually exercises the fork() path in
AgentKVBlockManager.allocate(), since vLLM builds one SequenceGroup with N
sequences forked from a shared parent for parallel sampling). This is a
reasonable proxy for both "many concurrent users sharing a system prompt"
and "one user's request explored via N branches" (e.g. best-of-N, ToT).

Version note
------------
integrations/vllm/block_manager.py was written against the exact vLLM
version vendored for reference in vllm-src/ — v0.4.3 (2024-05-31). That
exact release is no longer installable (its pinned `vllm-flash-attn==
2.5.8.post2` dependency has since been removed from PyPI - only 2.6.x
remains). v0.5.4 is the fallback: same block-manager era, likely resolvable
dependency pins:
    pip install vllm==0.5.4
If even that fails to resolve, try the next few 0.5.x/0.6.x releases in
order - vLLM's internals move fast, and a *much* newer release may have
removed or changed `BlockSpaceManager` / `get_block_space_manager_class`
entirely (vLLM's block-manager architecture has been reworked more than
once, including a full V1 engine rewrite). If `--backend agentkv` fails to
even patch on whatever version you land on, that's the first thing to
check - not necessarily a bug in AgentKV itself.

Usage
-----
# Sanity check first - tiny model, tiny workload, confirm the patch works
# inside a real engine at all before trusting a bigger comparison:
python bench/vllm_real_user_sim.py --model facebook/opt-125m \\
    --backend agentkv --n-users 2 --n-agents-per-user 2 --max-tokens 20

# Real comparison - run BOTH backends as separate processes on the same
# workload (fresh CUDA context each):
python bench/vllm_real_user_sim.py --model <model> --backend agentkv \\
    --n-users 6 --n-agents-per-user 4 --max-tokens 60
python bench/vllm_real_user_sim.py --model <model> --backend stock \\
    --n-users 6 --n-agents-per-user 4 --max-tokens 60
"""

import argparse
import os
import sys
import time

import torch

# `integrations` is only importable from the project root - pyproject.toml's
# editable install packages `agentkv*` only, not `integrations`, and
# `python bench/this_script.py` puts bench/'s own directory on sys.path[0],
# not the project root one level up. Add it explicitly rather than relying
# on the caller's cwd or a PYTHONPATH env var.
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
    "SUPPORT ESCALATION: Tier 1 support handles account, billing, and "
    "basic configuration questions. Tier 2 handles performance and "
    "integration issues. Security-related reports must always be escalated "
    "immediately to the security response team regardless of severity "
    "classification.\n\n"
    "Answer customer questions using only the above context where relevant, "
    "citing the incident date if referencing an outage, and clearly state "
    "when a question requires escalation rather than guessing."
)

USER_QUESTIONS = [
    "Was my account affected by any outage in the last two months?",
    "How is my service credit calculated for the June incident?",
    "Why did I see 503 errors on the API gateway in early July?",
    "Who do I contact if I think I found a security vulnerability?",
    "What's the difference between Tier 1 and Tier 2 support?",
    "Can you explain the root cause of the June 14th latency issue?",
    "Do I need to request a credit manually or is it automatic?",
    "Is the legacy API gateway issue fully resolved now?",
]


def build_prompts(n_users: int) -> list:
    questions = (USER_QUESTIONS * ((n_users // len(USER_QUESTIONS)) + 1))[:n_users]
    return [f"{SHARED_CONTEXT}\n\nCustomer question: {q}\nAnswer:" for q in questions]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--backend", choices=["agentkv", "stock"], default="agentkv",
                         help="agentkv = patch_vllm() applied; stock = unmodified vLLM BlockSpaceManagerV2")
    parser.add_argument("--n-users", type=int, default=4,
                         help="Number of distinct concurrent user questions submitted together")
    parser.add_argument("--n-agents-per-user", type=int, default=2,
                         help="Parallel completions per user question (SamplingParams(n=...)) - this is what forks")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--quantization", type=str, default=None,
                         help="e.g. 'awq', 'gptq' - only if --model is pre-quantized for one of these")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                         help="Skip CUDA graph capture - safer for a first real run, easier to debug on failure")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(f"AgentKV real-vLLM-engine simulation - backend={args.backend}")
    print(f"  model              = {args.model}")
    print(f"  n_users            = {args.n_users}")
    print(f"  n_agents_per_user  = {args.n_agents_per_user}")
    print(f"  max_tokens         = {args.max_tokens}")
    print(f"  total sequences    = {args.n_users * args.n_agents_per_user}")
    print("=" * 65)

    import vllm
    from vllm import SamplingParams

    if args.backend == "agentkv":
        from integrations.vllm import patch_vllm
        patch_vllm()
    else:
        print("\n[stock] Running with vLLM's own unmodified BlockSpaceManagerV2 - no patch applied.")

    print(f"\n[1/3] Loading {args.model} into vLLM LLMEngine...")
    t0 = time.perf_counter()
    llm_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = vllm.LLM(**llm_kwargs)
    load_s = time.perf_counter() - t0
    print(f"   Loaded in {load_s:.1f}s")

    prompts = build_prompts(args.n_users)
    sampling_params = SamplingParams(
        n=args.n_agents_per_user,
        max_tokens=args.max_tokens,
        temperature=0.0 if args.n_agents_per_user == 1 else 0.8,
        top_p=1.0 if args.n_agents_per_user == 1 else 0.95,
    )

    print(f"\n[2/3] Generating for {args.n_users} concurrent users "
          f"x {args.n_agents_per_user} completions each...")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e9

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    gen_s = time.perf_counter() - t0

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1e9
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    total_completions = 0
    total_new_tokens = 0
    for req_idx, out in enumerate(outputs):
        for comp_idx, completion in enumerate(out.outputs):
            total_completions += 1
            total_new_tokens += len(completion.token_ids)
            if req_idx == 0 and comp_idx == 0:
                print(f"   [sample output] user 1, completion 1: "
                      f"\"{completion.text[:80].strip()}\"")

    throughput = total_new_tokens / gen_s if gen_s > 0 else 0.0

    print(f"\n[3/3] Real measured results\n{'='*65}")
    print(f"  Backend                : {args.backend}")
    print(f"  Total sequences         : {total_completions} "
          f"({args.n_users} users x {args.n_agents_per_user} completions)")
    print(f"  Total new tokens        : {total_new_tokens}")
    print(f"  Generation wall time    : {gen_s:.2f} s")
    print(f"  Throughput              : {throughput:.1f} tokens/sec")
    print(f"  GPU mem before generate : {mem_before:.3f} GB")
    print(f"  GPU mem after generate  : {mem_after:.3f} GB")
    print(f"  Peak GPU mem during gen : {peak_mem:.3f} GB")
    print(f"\nDone - backend={args.backend}. Compare against the same command "
          f"with --backend {'stock' if args.backend == 'agentkv' else 'agentkv'} "
          f"for the real head-to-head numbers.")


if __name__ == "__main__":
    main()
