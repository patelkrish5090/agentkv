"""
vLLM Integration for AgentKV

This module patches vLLM to use AgentKVBlockManager as its default block allocator.
"""
from typing import Any
import importlib

def patch_vllm() -> None:
    """
    Patches vLLM to use AgentKVBlockManager instead of BlockSpaceManagerV2.
    Must be called before initializing the vLLM LLM or AsyncLLMEngine.
    """
    try:
        import vllm.core.interfaces
    except ImportError:
        raise ImportError("vLLM is not installed. Please install vllm>=0.4.0.")

    from .block_manager import AgentKVBlockManager

    # Store the original just in case
    original_get_class = vllm.core.interfaces.BlockSpaceManager.get_block_space_manager_class

    def patched_get_class(version: str) -> Any:
        if version.lower() == "v2":
            return AgentKVBlockManager
        return original_get_class(version)

    # Apply the monkey patch
    vllm.core.interfaces.BlockSpaceManager.get_block_space_manager_class = staticmethod(patched_get_class)
    
    print("AgentKV: Successfully patched vLLM to use AgentKVBlockManager!")

__all__ = ["patch_vllm", "AgentKVBlockManager"]
