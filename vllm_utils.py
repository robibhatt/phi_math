from vllm import LLM, SamplingParams, TokensPrompt
from typing import Union, List, Dict, Any
import numpy as np
from dataclasses import dataclass

def get_llm(llm :Union[str, LLM], **kwargs) -> LLM:
    if isinstance(llm, str):
        return LLM(model=llm, **kwargs)
    return llm


@dataclass
class VllmModelConfig:
    # ---- Model / runtime ----
    model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # vLLM expects 'dtype' (torch_dtype is deprecated)
    dtype: str  # e.g., "half", "bfloat16", "float32", "auto"

    # ---- Generation / decoding ----
    max_tokens: int
    temperature: float
    batch_size: int


def init_model(cfg: VllmModelConfig):
    """
    Initialize and return the LLM with the provided ModelConfig.
    """
    return get_llm(
        cfg.model_name,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,  # correct: use 'dtype' (not torch_dtype)
    )


def get_generation_likelihood(
    llm: LLM,
    prompts: List[str],
    **generation_kwargs
) -> List[Dict[str, Any]]:
    """
    Generate text and return likelihoods (sum of sampled-token logprobs).
    Returns entries with: 'text', 'token_ids', 'likelihood'.
    Fails fast if any sampled-token logprob is missing or shapes are inconsistent.
    """

    # Ensure we request per-token logprobs (>=1 aims to include the sampled token)
    generation_kwargs["logprobs"] = max(1, int(generation_kwargs.get("logprobs", 1)))

    sampling_params = SamplingParams(**generation_kwargs)
    outputs = llm.generate(prompts, sampling_params)

    results: List[Dict[str, Any]] = []
    for output in outputs:
        generated = output.outputs[0]
        token_ids = generated.token_ids 
        text = generated.text
        lp_dicts = generated.logprobs

        # Require that we actually generated something
        if not token_ids:
            raise RuntimeError("No generated token IDs returned; cannot compute likelihood.")

        # Shape alignment must hold
        if lp_dicts is None:
            raise RuntimeError("No per-token logprobs returned; set a positive `logprobs`.")
        if len(token_ids) != len(lp_dicts):
            raise RuntimeError(
                f"token_ids ({len(token_ids)}) and logprobs ({len(lp_dicts)}) lengths differ."
            )

        token_logprobs: List[float] = []
        for tid, lp_dict in zip(token_ids, lp_dicts):
            if not lp_dict:
                raise RuntimeError("Empty logprob dict for a generated token position.")

            # Find the sampled token's logprob at this position (handle key type variants)
            info = lp_dict.get(tid) or lp_dict.get(int(tid)) or lp_dict.get(str(tid))
            if info is None:
                raise KeyError(f"Sampled token id {tid} missing from logprob dict.")
            token_logprobs.append(info.logprob)

        likelihood = sum(token_logprobs)

        results.append({
            "text": text,
            "token_ids": token_ids,
            "likelihood": likelihood,
        })

    return results