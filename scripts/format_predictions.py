#!/usr/bin/env python
"""Convert predictions.jsonl to human-readable predictions.txt"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local src/ is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    src_str = str(SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from phi_synth_math.tasks.core.predictions_formatter import format_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert predictions.jsonl to human-readable predictions.txt"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to predictions.jsonl file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path (default: predictions.txt in same directory)",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = args.output.expanduser().resolve()
    else:
        output_path = input_path.parent / "predictions.txt"

    format_predictions(input_path, output_path)

    # Print summary
    n_total = 0
    n_correct = 0
    import json
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                n_total += 1
                if record.get("correct", False):
                    n_correct += 1
    accuracy = (n_correct / n_total * 100) if n_total > 0 else 0.0
    print(f"Formatted {n_total} predictions -> {output_path}")
    print(f"Accuracy: {n_correct}/{n_total} ({accuracy:.1f}%)")


if __name__ == "__main__":
    main()
