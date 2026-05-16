"""
Standalone Interlat training-data collector.

Reads agent personas from the OASIS CSV, generates a tweet response to
the source post using the local HuggingFace model, and captures the
last K hidden states — all in a single forward pass per agent.

Output:
  <out_dir>/conversations.jsonl   — one record per agent per round
  <out_dir>/hidden/<id>.pt        — sender hidden states (K, d_h)

Usage:
    python data_collection/run_collection_standalone.py \
        --csv data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv \
        --source-tweet "report: amazon plans to open its first physical store, in new york" \
        --model Qwen/Qwen3-4B \
        --K 8 \
        --rounds 3 \
        --output runs/false_business_0_collect
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# resolve OASIS user.py system prompt format
OASIS_ROOT = Path(__file__).resolve().parents[2] / "oasis"
sys.path.insert(0, str(OASIS_ROOT))


def build_system_prompt(row: pd.Series) -> str:
    name = row.get("name", "user")
    profile = row.get("user_char", row.get("description", ""))
    return (
        "# OBJECTIVE\n"
        "You're a Twitter user, and I'll present you with some tweets. "
        "After you see the tweets, write a single tweet of your own in response "
        "(under 280 characters). Output only the tweet text, nothing else.\n\n"
        "# SELF-DESCRIPTION\n"
        f"Your name is {name}.\n"
        f"Your profile: {profile}."
    )


def build_user_prompt(source_tweet: str, round_idx: int) -> str:
    return (
        f"Here is your social media environment:\n"
        f"Posts: [{{'content': '{source_tweet}', 'likes': 0, 'reposts': 0}}]\n\n"
        f"Write a tweet in response. Round {round_idx + 1}."
    )


def build_input_ids(tokenizer, system: str, user: str, device):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--source-tweet",
                        default="report: amazon plans to open its first physical store, in new york")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3,
                        help="How many prompt variations per agent")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", default="runs/false_business_0_collect")
    args = parser.parse_args()

    out_dir = Path(args.output)
    hidden_dir = out_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "conversations.jsonl"

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = next(model.parameters()).device

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} agents from {args.csv}")

    total = 0
    with open(jsonl_path, "a") as log:
        for _, row in df.iterrows():
            agent_id = int(row.get("user_id", row.name))
            system = build_system_prompt(row)

            for r in range(args.rounds):
                uid = f"agent{agent_id}_r{r:02d}"
                pt_path = hidden_dir / f"{uid}.pt"
                if pt_path.exists():
                    continue

                user = build_user_prompt(args.source_tweet, r)
                inputs = build_input_ids(tokenizer, system, user, device)

                with torch.no_grad():
                    # Capture hidden states and generate in one call
                    hidden_outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    sender_states = hidden_outputs.hidden_states[-1][0, -args.K:, :].cpu()

                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                new_tokens = gen_out[0, inputs["input_ids"].shape[1]:]
                post_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                torch.save({"id": uid, "hidden": sender_states}, pt_path)

                record = {
                    "id": uid,
                    "agent_id": agent_id,
                    "round": r,
                    "system": system,
                    "user": user,
                    "post": post_text,
                    "hidden_state": str(pt_path.resolve()),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                total += 1

            if (int(row.name) + 1) % 10 == 0:
                print(f"  {int(row.name)+1}/{len(df)} agents, {total} records")

    print(f"\nDone. {total} records → {jsonl_path}")
    print(f"Hidden states → {hidden_dir}/")


if __name__ == "__main__":
    main()
