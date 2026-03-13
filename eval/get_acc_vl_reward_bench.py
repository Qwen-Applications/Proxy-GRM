import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


CATEGORIES = ("reasoning", "hallucination", "general")


def classify_id(data_id: str) -> str:
    s = data_id
    if ("mmmu" in s) or ("mathverse" in s):
        return "reasoning"
    if ("hallucination" in s) or ("rlhf" in s) or ("rlaif" in s):
        return "hallucination"
    return "general"


@dataclass
class Stat:
    count: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.count if self.count else 0.0


def calculate_accuracy_and_stats(file_path: str) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    stats: Dict[str, Stat] = {c: Stat() for c in CATEGORIES}

    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                data_id = data["id"]
                rewards = data["rewards"]
                reverse = data["reverse"]
            except Exception as e:
                raise ValueError(f"Bad jsonl line {lineno}: {e}\nLINE={line[:200]}") from e

            if not (isinstance(rewards, (list, tuple)) and len(rewards) >= 2):
                raise ValueError(f"Line {lineno}: rewards must have at least 2 elements, got {rewards}")

            category = classify_id(str(data_id))
            st = stats[category]
            st.count += 1

            r0, r1 = rewards[0], rewards[1]
            is_correct = (r0 > r1) if not reverse else (r0 < r1)
            st.correct += int(is_correct)

    results = {
        c: {"samples": stats[c].count, "accuracy": stats[c].accuracy}
        for c in CATEGORIES
    }

    total_samples = sum(stats[c].count for c in CATEGORIES)
    total_correct = sum(stats[c].correct for c in CATEGORIES)
    micro_acc = total_correct / total_samples if total_samples else 0.0  # overall accuracy
    macro_acc = sum(results[c]["accuracy"] for c in CATEGORIES) / len(CATEGORIES) 

    return results, micro_acc, macro_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to .jsonl file")
    args = parser.parse_args()

    results, micro_acc, macro_acc = calculate_accuracy_and_stats(args.file_path)

    input_path = Path(args.file_path)
    output_path = input_path.with_name(input_path.stem + "_acc.txt")

    lines = []
    for category in CATEGORIES:
        s = results[category]
        lines.append(f"Category: {category}")
        lines.append(f"  Samples: {s['samples']}")
        lines.append(f"  Accuracy: {s['accuracy']:.4f}")

    lines.append(f"Average Accuracy (Micro): {micro_acc:.4f}")
    lines.append(f"Macro Accuracy: {macro_acc:.4f}")

    print("\n".join(lines))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
