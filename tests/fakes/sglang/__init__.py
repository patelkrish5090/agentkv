"""Minimal fake ``sglang`` package used only in tests.

Real SGLang cannot be installed in this dev environment (it requires a CUDA
build). This stub reproduces the subset of ``sglang.srt.mem_cache`` that
``integrations/sglang/radix_cache.py`` implements against — trimmed to the
fields/methods that adapter actually touches. Signatures were pulled from
SGLang's `main` branch (commit 10908a67931872d96691f41d1f91f6f0c3f4fded,
2026-07-18) — see integrations/sglang/README.md for the version-pin caveat;
SGLang's mem_cache API moves fast and these stubs will drift from real
SGLang releases over time.

Added to sys.path by individual test modules — real SGLang, if ever
installed, always wins because it would already be in sys.modules.
"""
