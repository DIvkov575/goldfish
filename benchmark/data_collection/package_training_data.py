"""
Phase C: merge JSONL conversation logs + .pt hidden states into the
training format expected by train_adapter.py.

Usage:
    python package_training_data.py \
        --jsonl runs/false_business_0.jsonl \
        --hidden runs/false_business_0_hidden/ \
        --output data/oasis_interlat_train.json \
        --eval-split 0.1
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--hidden", required=True, help="Directory of .pt files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hidden_dir = Path(args.hidden)

    rows = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    records = []
    missing = 0
    for i, row in enumerate(rows):
        # Support both standalone format (has hidden_state path) and log_hook format
        if "hidden_state" in row:
            pt_path = Path(row["hidden_state"])
        else:
            uid = f"agent{row['agent_id']}_{i:05d}"
            pt_path = hidden_dir / f"{uid}.pt"

        if not pt_path.exists():
            missing += 1
            continue

        records.append({
            "id": row.get("id", f"agent{row['agent_id']}_{i:05d}"),
            "conversations": [
                {"from": "human", "value": row["user"]},
                {"from": "gpt",   "value": row["post"]},
            ],
            "system": row["system"],
            "hidden_state": str(pt_path.resolve()),
        })

    if missing:
        print(f"Warning: {missing} rows missing hidden state files, skipped.")

    random.seed(args.seed)
    random.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_split))
    train = records[n_eval:]
    eval_ = records[:n_eval]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump({"train": train, "eval": eval_}, f, indent=2)

    print(f"Packaged {len(train)} train + {len(eval_)} eval records → {out}")


if __name__ == "__main__":
    main()
