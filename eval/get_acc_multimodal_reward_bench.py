import argparse
import json
from collections import defaultdict
from pathlib import Path


def normalize_category(ex: dict) -> str:
    category = str(ex.get("Category", "unknown")).lower()
    ex_id = str(ex.get("id", "")).lower()

    if category == "safety":
        return "safety/bias" if ex_id.startswith("pairs") else "safety/toxicity"
    if category == "reasoning":
        return "reasoning/math" if ex_id.startswith("math") else "reasoning/coding"
    return category


def is_correct(ex: dict) -> int:
    rewards = ex.get("rewards")
    reverse = bool(ex.get("reverse", False))

    if not isinstance(rewards, (list, tuple)) or len(rewards) < 2:
        raise ValueError(f"Invalid rewards: {rewards}")

    r0, r1 = rewards[0], rewards[1]
    return int((r0 > r1) if not reverse else (r0 < r1))


def compute_acc(file_path: str) -> str:
    accs = defaultdict(list)

    total_correct = 0
    total_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad json at line {lineno}: {e}")

            acc = is_correct(ex)
            cat = normalize_category(ex)

            accs["all"].append(acc)
            accs[cat].append(acc)

            total_correct += acc
            total_count += 1

    # 输出：all 放第一，其它按名字排序，方便对比
    keys = ["all"] + sorted(k for k in accs.keys() if k != "all")

    out_path = str(Path(file_path).with_name(Path(file_path).stem + "_results.txt"))
    with open(out_path, "w", encoding="utf-8") as w:
        for k in keys:
            c = sum(accs[k])
            n = len(accs[k])
            acc = c / n if n else 0.0
            line = f"acc {k}: {c} / {n} = {acc:.6f}"
            print(line)
            w.write(line + "\n")

        micro = total_correct / total_count if total_count else 0.0
        macro = sum((sum(accs[k]) / len(accs[k])) for k in keys if len(accs[k]) > 0) / sum(1 for k in keys if len(accs[k]) > 0)

        w.write(f"Overall (Micro): {total_correct} / {total_count} = {micro:.6f}\n")
        w.write(f"Overall (Macro over shown groups): {macro:.6f}\n")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to .jsonl file")
    args = parser.parse_args()
    compute_acc(args.file_path)


if __name__ == "__main__":
    main()
