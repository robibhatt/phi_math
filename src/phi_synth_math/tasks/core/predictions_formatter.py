"""Format predictions.jsonl to human-readable predictions.txt"""

from __future__ import annotations

import json
from pathlib import Path


def format_predictions(jsonl_path: Path, output_path: Path) -> None:
    """Convert predictions.jsonl to formatted txt."""
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_total = len(records)
    n_correct = sum(1 for r in records if r.get("correct", False))
    accuracy = (n_correct / n_total * 100) if n_total > 0 else 0.0

    sep = "=" * 80

    with output_path.open("w", encoding="utf-8") as out:
        # Summary header
        out.write(sep + "\n")
        out.write(f"EVALUATION RESULTS: {n_correct}/{n_total} correct ({accuracy:.1f}%)\n")
        out.write(f"Source: {jsonl_path}\n")
        out.write(sep + "\n\n")

        for i, record in enumerate(records, start=1):
            ex_id = record.get("id", f"example_{i}")
            correct = record.get("correct", False)
            status = "CORRECT" if correct else "INCORRECT"

            # Header line with ID and status
            header = f"[{i}] {ex_id}"
            padding = 80 - len(header) - len(status) - 1
            if padding < 1:
                padding = 1

            out.write(sep + "\n")
            out.write(f"{header}{' ' * padding}{status}\n")
            out.write(sep + "\n\n")

            # Problem/question
            question = record.get("question", "")
            out.write("PROBLEM:\n")
            out.write(question.strip() + "\n\n")

            # Expected answer
            gold = record.get("gold", "")
            out.write("EXPECTED ANSWER:\n")
            out.write(str(gold).strip() + "\n\n")

            # Model response
            pred = record.get("pred", "")
            out.write("MODEL RESPONSE:\n")
            out.write(pred.strip() + "\n\n")
