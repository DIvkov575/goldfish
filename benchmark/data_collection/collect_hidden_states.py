"""
Phase B: extract Interlat sender hidden states from logged conversation pairs.

Reads the JSONL written by log_hook, builds the exact same chat-template
prompt the Interlat sender uses, runs the local HuggingFace model with
output_hidden_states=True, and saves the last K hidden states per row.

Usage:
    python collect_hidden_states.py \
        --input runs/false_business_0.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --K 8 \
        --output runs/false_business_0_hidden/
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, system: str, user: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL from log_hook")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--output", required=True, help="Directory for .pt files")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = next(model.parameters()).device

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Processing {len(rows)} rows ...")

    for i, row in enumerate(rows):
        uid = f"agent{row['agent_id']}_{i:05d}"
        out_path = out_dir / f"{uid}.pt"

        if out_path.exists():
            continue

        inputs = build_prompt(tokenizer, row["system"], row["user"])
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        last_layer = outputs.hidden_states[-1]          # (1, seq_len, d_h)
        sender_states = last_layer[0, -args.K:, :].cpu()  # (K, d_h)

        torch.save({"id": uid, "hidden": sender_states}, out_path)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}")

    print(f"Done. Saved to {out_dir}/")


if __name__ == "__main__":
    main()
