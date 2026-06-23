# SGLang Integration for AgentKV

## Overview

SGLang introduced **RadixAttention**, which fundamentally shares the same architectural goals as AgentKV's `DualRadixTree`: providing prefix caching and CoW (Copy-on-Write) caching for LLM inference natively at the cache manager level.

Since SGLang inherently handles sequence branching via RadixAttention and manages the VRAM blocks efficiently using its own epoch-based eviction tree, injecting AgentKV into SGLang's `RadixCache` is functionally redundant. 

SGLang users already get the O(1) branching and CoW benefits natively!

## Difference between AgentKV and SGLang RadixAttention
While SGLang's RadixAttention focuses primarily on a system-level HTTP inference server environment, **AgentKV** is designed to be a lightweight, library-agnostic memory pool. 

AgentKV allows developers to easily inject RadixTree CoW logic into:
- HuggingFace `transformers.Cache`
- vLLM `BlockSpaceManager`
- Custom PyTorch scripts and multi-agent training loops (via pure PyTorch implementations).

Therefore, there is no direct SGLang hook provided, as AgentKV and SGLang offer equivalent architectural advantages in their respective domains.
