import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


CATEGORY_KEYWORDS = ["mcq", "long", "short", "safety", "video"]


def get_media_path(item: dict) -> str:
    return (item.get("image") or item.get("video") or "").lower()


def match_category(media_path: str, keywords=CATEGORY_KEYWORDS) -> Optional[str]:
    for kw in keywords:
        if kw in media_path:
            return kw
    return None


def is_correct(item: dict) -> bool:
    rewards = item.get("rewards", None)
    reverse = bool(item.get("reverse", False))
    if not isinstance(rewards, (list, tuple)) or len(rewards) < 2:
        raise ValueError(f"Invalid rewards: {rewards}")
    r0, r1 = rewards[0], rewards[1]
    return (r0 > r1) if not reverse else (r0 < r1)


@dataclass
class Stats:

    total_samples: int = 0
    correct_samples: int = 0

    total_ids: int = 0
    all_correct_ids: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct_samples / self.total_samples if self.total_samples else 0.0

    @property
    def acc_plus(self) -> float:
        return self.all_correct_ids / self.total_ids if self.total_ids else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to .jsonl file")
    args = parser.parse_args()

    id_to_items: Dict[str, List[dict]] = defaultdict(list)
    id_to_cat: Dict[str, Optional[str]] = {}

    with open(args.file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad json line {lineno}: {e}")

            item_id = str(item.get("id", ""))
            id_to_items[item_id].append(item)

            if item_id not in id_to_cat:
                media = get_media_path(item)
                id_to_cat[item_id] = match_category(media)

    
    cat_stats: Dict[str, Stats] = {kw: Stats() for kw in CATEGORY_KEYWORDS}
    overall = Stats()

    for item_id, items in id_to_items.items():
        cat = id_to_cat.get(item_id)
        all_correct = True

        for it in items:
            correct = is_correct(it)

            overall.total_samples += 1
            overall.correct_samples += int(correct)

            if cat in cat_stats:
                cat_stats[cat].total_samples += 1
                cat_stats[cat].correct_samples += int(correct)

            if not correct:
                all_correct = False

        overall.total_ids += 1
        overall.all_correct_ids += int(all_correct)

        if cat in cat_stats:
            cat_stats[cat].total_ids += 1
            cat_stats[cat].all_correct_ids += int(all_correct)

    output_path = str(Path(args.file_path).with_suffix("").as_posix() + "_result.txt")
    lines = []
    lines.append("Category-wise Metrics:")
    for kw in CATEGORY_KEYWORDS:
        st = cat_stats[kw]
        lines.append(f"Category: {kw}")
        lines.append(f"  Accuracy: {st.accuracy:.4f}")
        lines.append(f"  ACC+: {st.acc_plus:.4f}")
        lines.append(f"  Total Samples: {st.total_samples}")
        lines.append(f"  Total IDs: {st.total_ids}")

    lines.append("")
    lines.append("Overall Metrics:")
    lines.append(f"Overall Accuracy: {overall.accuracy:.4f}")
    lines.append(f"Overall ACC+: {overall.acc_plus:.4f}")
    lines.append(f"Total Samples: {overall.total_samples}")
    lines.append(f"Total IDs: {overall.total_ids}")

    print("\n".join(lines))
    with open(output_path, "w", encoding="utf-8") as w:
        w.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()