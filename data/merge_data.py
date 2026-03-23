"""
merge_data.py
Merges all topic JSONL files, shuffles, and splits into train/validation sets.
Run this after all convert_*.py scripts.
"""

import json
import random
import os

FILES = [
    ("train_nutrition.jsonl", "Nutrition"),
    ("train_growing.jsonl",   "Growing"),
    ("train_chemicals.jsonl", "Chemicals"),
    ("train_diseases.jsonl",  "Diseases"),
]

def merge(val_split=0.1, seed=42):
    random.seed(seed)

    all_pairs = []
    counts    = {}

    print("Loading files...\n")
    for filename, label in FILES:
        if not os.path.exists(filename):
            print(f"  ⚠️  {filename} not found — skipping")
            continue
        with open(filename) as f:
            lines = [l.strip() for l in f if l.strip()]
        counts[label] = len(lines)
        all_pairs.extend(lines)
        print(f"  ✓  {label}: {len(lines)} pairs")

    print(f"\nTotal before split: {len(all_pairs)} pairs")

    random.shuffle(all_pairs)

    split     = int(len(all_pairs) * (1 - val_split))
    train_set = all_pairs[:split]
    val_set   = all_pairs[split:]

    with open("train_final.jsonl", "w") as f:
        f.write("\n".join(train_set) + "\n")

    with open("val_final.jsonl", "w") as f:
        f.write("\n".join(val_set) + "\n")

    print(f"\n{'─'*40}")
    print(f"  Training:   {len(train_set)} pairs  → train_final.jsonl")
    print(f"  Validation: {len(val_set)} pairs  → val_final.jsonl")
    print(f"{'─'*40}")
    print("\nBreakdown by topic:")
    for label, count in counts.items():
        pct = round(count / len(all_pairs) * 100, 1)
        print(f"  {label}: {count} ({pct}%)")

    print("\n✅ Done. Upload train_final.jsonl and val_final.jsonl to Google Colab.")

    # Validate a sample
    print("\nValidating sample...")
    try:
        sample = json.loads(train_set[0])
        assert "messages" in sample
        assert len(sample["messages"]) == 3
        print("  ✓ Format looks correct")
    except Exception as e:
        print(f"  ✗ Format issue: {e}")

if __name__ == "__main__":
    merge()
