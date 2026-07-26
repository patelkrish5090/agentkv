"""Minimal fake ``vllm`` package used only in tests.

Real vLLM cannot be installed in this dev environment (it requires Triton
and a CUDA build, neither available on native Windows CI). Importing the
real package also pulls in its full engine/config/model_executor stack just
to reach the two small modules our integration actually depends on
(``vllm.core.interfaces`` and ``vllm.sequence``).

This stub reproduces only the subset of that surface AgentKVBlockManager
uses, with signatures copied from the vendored reference in ``vllm-src/``
(see ``vllm-src/vllm/core/interfaces.py`` and ``vllm-src/vllm/sequence.py``).
It is added to ``sys.path`` by ``tests/conftest.py`` — real vLLM, if ever
installed, always wins because it would already be in ``sys.modules``.
"""
