from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from vllm import LLM, SamplingParams


# ---------------------------
# IO: queries
# ---------------------------

def read_queries_txt(path: Path, *, mode: str = "paragraphs") -> List[str]:
    """
    Supported modes:
      - "paragraphs": queries are paragraphs separated by one or more blank lines
      - "lines": each non-empty, non-comment line is a query

    Comments: lines whose first non-space char is '#'
    """
    text = path.read_text(encoding="utf-8")
    raw_lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        raw_lines.append(line.rstrip("\n"))

    cleaned = "\n".join(raw_lines).strip()
    if not cleaned:
        return []

    if mode == "lines":
        queries = [ln.strip() for ln in cleaned.splitlines() if ln.strip() != ""]
        return queries

    if mode != "paragraphs":
        raise ValueError(f"Unsupported queries.mode={mode!r}. Use 'paragraphs' or 'lines'.")

    # paragraphs
    chunks: List[str] = []
    buf: List[str] = []
    for line in cleaned.splitlines():
        if line.strip() == "":
            if buf:
                chunks.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line.rstrip())
    if buf:
        chunks.append("\n".join(buf).strip())

    return [c for c in chunks if c.strip()]


# ---------------------------
# IO: output formatting
# ---------------------------

def write_outputs_txt(
    out_path: Path,
    *,
    header: Dict[str, Any],
    prompts: List[str],
    completions: List[str],
) -> None:
    sep = "=" * 88
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("vLLM batch completions\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # pretty header block
        for k, v in header.items():
            f.write(f"{k}: {v}\n")

        f.write(f"Num queries: {len(prompts)}\n")
        f.write(f"{sep}\n\n")

        for i, (p, c) in enumerate(zip(prompts, completions), start=1):
            f.write(f"[{i}/{len(prompts)}]\n")
            f.write("-" * 88 + "\n")
            f.write("PROMPT:\n")
            f.write(p.rstrip() + "\n\n")
            f.write("COMPLETION:\n")
            f.write(c.rstrip() + "\n")
            f.write("\n" + sep + "\n\n")


# ---------------------------
# YAML config loading
# ---------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping/dict.")
    return data


def require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    # Convention: config.yaml next to script or passed via env/you can edit below.
    # To keep it dead simple: you edit CONFIG_PATH in this file.
    CONFIG_PATH = Path("run_config.yaml")

    cfg = load_yaml(CONFIG_PATH)

    model_cfg = require(cfg, "model")
    io_cfg = require(cfg, "io")
    sampling_cfg = require(cfg, "sampling")

    model_name = require(model_cfg, "name")

    # IO
    queries_path = Path(require(io_cfg, "queries_file"))
    out_path = Path(require(io_cfg, "output_file"))
    queries_mode = io_cfg.get("queries_mode", "paragraphs")

    # runtime model init
    llm = LLM(
        model=model_name,
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.90)),
        max_model_len=int(model_cfg.get("max_model_len", 4096)),
        dtype=str(model_cfg.get("dtype", "auto")),
        # passthrough for any extra vLLM LLM kwargs if you want:
        **(model_cfg.get("extra_llm_kwargs", {}) or {}),
    )

    prompts = read_queries_txt(queries_path, mode=queries_mode)
    if not prompts:
        raise SystemExit(f"No queries found in {queries_path} (after stripping comments/blank lines).")

    sp = SamplingParams(
        max_tokens=int(sampling_cfg.get("max_tokens", 256)),
        temperature=float(sampling_cfg.get("temperature", 0.7)),
        top_p=float(sampling_cfg.get("top_p", 1.0)),
        seed=sampling_cfg.get("seed", None),
        stop=sampling_cfg.get("stop", None),
        n=int(sampling_cfg.get("n", 1)),
        **(sampling_cfg.get("extra_sampling_kwargs", {}) or {}),
    )

    outputs = llm.generate(prompts, sp)

    # If n>1, we’ll write all completions per prompt (nicely concatenated).
    completions: List[str] = []
    n = int(sampling_cfg.get("n", 1))

    for out in outputs:
        if n == 1:
            completions.append(out.outputs[0].text)
        else:
            parts = []
            for j, comp in enumerate(out.outputs, start=1):
                parts.append(f"(completion {j}/{n})\n{comp.text.rstrip()}")
            completions.append("\n\n".join(parts))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "Model": model_name,
        "Queries file": str(queries_path),
        "Output file": str(out_path),
        "Queries mode": queries_mode,
        "Sampling": {
            "max_tokens": sp.max_tokens,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "seed": sampling_cfg.get("seed", None),
            "stop": sampling_cfg.get("stop", None),
            "n": n,
        },
        "Runtime": {
            "tensor_parallel_size": int(model_cfg.get("tensor_parallel_size", 1)),
            "gpu_memory_utilization": float(model_cfg.get("gpu_memory_utilization", 0.90)),
            "max_model_len": int(model_cfg.get("max_model_len", 4096)),
            "dtype": str(model_cfg.get("dtype", "auto")),
        },
    }

    write_outputs_txt(out_path, header=header, prompts=prompts, completions=completions)
    print(f"Wrote {len(prompts)} completions -> {out_path}")


if __name__ == "__main__":
    main()
