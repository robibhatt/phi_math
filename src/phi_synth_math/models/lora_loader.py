"""Load trained LoRA adapters for inference."""

from __future__ import annotations

import json
from pathlib import Path

from phi_synth_math.models.base import Model
from phi_synth_math.models.dummy import DummyModel


def load_lora_model(
    base_model: str,
    adapter_path: Path,
    device: str = "cpu",
) -> Model:
    """Load base model with LoRA adapter for inference.

    Args:
        base_model: Name or path of the base model. Use "dummy" for testing.
        adapter_path: Path to the saved adapter directory.
        device: Device to load model on (e.g., "cpu", "cuda").

    Returns:
        Model instance ready for inference.

    Raises:
        FileNotFoundError: If adapter_path doesn't exist.
        ValueError: If adapter_config.json is missing from adapter_path.
    """
    adapter_path = Path(adapter_path)

    # Validate adapter path exists
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    # Validate adapter_config.json exists
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise ValueError(
            f"adapter_config.json not found in {adapter_path}. "
            "Expected a valid PEFT adapter directory."
        )

    # Load adapter config
    with config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    # For testing: return DummyModel when base_model is "dummy"
    if base_model == "dummy":
        return DummyModel()

    # For real models: use PEFT to load adapter
    # This requires transformers and peft packages
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Loading real LoRA models requires 'transformers' and 'peft' packages. "
            "Install with: pip install transformers peft"
        ) from e

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device if device != "cpu" else None,
        torch_dtype="auto",
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    if device == "cpu":
        model = model.to("cpu")

    # Wrap in a class that satisfies Model protocol
    return _LoRAModelWrapper(model, tokenizer)


class _LoRAModelWrapper:
    """Wrapper to make PEFT model satisfy the Model protocol."""

    def __init__(self, model, tokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model.eval()

    def generate(
        self,
        questions: list[str],
        *,
        max_tokens: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of questions.

        Args:
            questions: List of input prompts.
            max_tokens: Maximum tokens to generate per response.

        Returns:
            List of generated responses.
        """
        import torch

        if max_tokens is None:
            max_tokens = 256

        outputs = []
        with torch.no_grad():
            for question in questions:
                inputs = self._tokenizer(
                    question,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

                # Decode only the new tokens (exclude input)
                input_len = inputs["input_ids"].shape[1]
                response = self._tokenizer.decode(
                    generated[0][input_len:],
                    skip_special_tokens=True,
                )
                outputs.append(response)

        return outputs
