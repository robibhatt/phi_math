from __future__ import annotations

from typing import List

from vllm import LLM, SamplingParams

from .base import Model


"""
os.environ.setdefault("VLLM_USE_V1", "0")  # force legacy engine (usually most stable)
os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")  # avoid custom all-reduce kernel
os.environ.setdefault("VLLM_USE_CUDA_GRAPH", "0")  # disable cudagraph capture
os.environ.setdefault("VLLM_ENFORCE_EAGER", "1")  # avoid torch.compile paths
"""


class VLLMModel(Model):
    """vLLM-backed model wrapper that supports batch generation."""

    _DEFAULT_MAX_TOKENS = 16

    def __init__(
        self,
        *,
        model_name: str | None = None,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        dtype: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> None:
        if model_name is None:
            raise ValueError("model_name must be provided for the vLLM model.")

        llm_kwargs: dict[str, object] = {}
        if tensor_parallel_size is not None:
            llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if dtype is not None:
            llm_kwargs["dtype"] = dtype

        self._llm = LLM(model=model_name, **llm_kwargs)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed

    def generate(self, questions: List[str], *, max_tokens: int | None = None) -> List[str]:
        max_tokens_value = (
            max_tokens
            if max_tokens is not None
            else self._max_tokens
            if self._max_tokens is not None
            else self._DEFAULT_MAX_TOKENS
        )

        sampling_kwargs: dict[str, object] = {"max_tokens": max_tokens_value}
        if self._temperature is not None:
            sampling_kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            sampling_kwargs["top_p"] = self._top_p
        if self._seed is not None:
            sampling_kwargs["seed"] = self._seed

        sampling_params = SamplingParams(**sampling_kwargs)

        outputs = self._llm.generate(questions, sampling_params)

        # vLLM generally preserves input order, but make ordering explicit by request_id.
        # In most vLLM versions, request_id is the prompt index as a string: "0", "1", ...
        by_id = {out.request_id: out for out in outputs}

        generations: List[str] = []
        for i in range(len(questions)):
            out = by_id.get(str(i))
            if out is None or not out.outputs:
                generations.append("")
            else:
                generations.append(out.outputs[0].text.strip())

        return generations
