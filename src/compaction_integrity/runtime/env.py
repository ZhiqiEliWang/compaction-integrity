import os
from dataclasses import dataclass
from typing import Any


DEFAULT_HF_HOME = "/data/huggingface_cache"
DEFAULT_VLLM_CACHE_ROOT = "/data/huggingface_cache"


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentConfig:
    hf_home: str = DEFAULT_HF_HOME
    vllm_cache_root: str = DEFAULT_VLLM_CACHE_ROOT
    vllm_worker_multiproc_method: str | None = None


def resolve_runtime_environment_config(
    config: Any = None,
    *,
    worker_multiproc_method: str | None = None,
) -> RuntimeEnvironmentConfig:
    if isinstance(config, RuntimeEnvironmentConfig):
        if worker_multiproc_method is None:
            return config
        return RuntimeEnvironmentConfig(
            hf_home=config.hf_home,
            vllm_cache_root=config.vllm_cache_root,
            vllm_worker_multiproc_method=worker_multiproc_method,
        )

    env_config = config
    if config is not None and hasattr(config, "get"):
        nested_env = config.get("env")
        if nested_env is not None:
            env_config = nested_env

    hf_home = DEFAULT_HF_HOME
    vllm_cache_root = DEFAULT_VLLM_CACHE_ROOT
    vllm_worker_value = worker_multiproc_method

    if env_config is not None and hasattr(env_config, "get"):
        hf_home = str(env_config.get("hf_home") or hf_home)
        vllm_cache_root = str(env_config.get("vllm_cache_root") or vllm_cache_root)
        configured_worker_value = env_config.get("vllm_worker_multiproc_method")
        if configured_worker_value is not None and worker_multiproc_method is None:
            vllm_worker_value = str(configured_worker_value)

    return RuntimeEnvironmentConfig(
        hf_home=hf_home,
        vllm_cache_root=vllm_cache_root,
        vllm_worker_multiproc_method=vllm_worker_value,
    )


def apply_runtime_environment(
    config: Any = None,
    *,
    worker_multiproc_method: str | None = None,
) -> RuntimeEnvironmentConfig:
    resolved = resolve_runtime_environment_config(
        config,
        worker_multiproc_method=worker_multiproc_method,
    )
    os.environ["HF_HOME"] = resolved.hf_home
    os.environ["VLLM_CACHE_ROOT"] = resolved.vllm_cache_root
    if resolved.vllm_worker_multiproc_method is not None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = (
            resolved.vllm_worker_multiproc_method
        )
    return resolved
