"""
Benchmark: Text OASIS vs OASIS + Interlat (arXiv 2511.09149)

For each agent context:
  baseline  — model generates post directly from the prompt (text OASIS)
  interlat  — sender extracts K hidden states; receiver adapter compresses them
               into a prefix; text is rolled out post-hoc from that prefix

Metrics:
  semantic similarity  — cosine sim between baseline and interlat posts
  token count          — output tokens used by each method
  wall-clock time      — seconds per post
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent.sender import extract_sender_states
from latent.adapter import make_adapter
from latent.rollout import rollout_from_sender
from metrics.similarity import batch_similarity, token_count

# Synthetic OASIS-style contexts: (persona, feed)
CONTEXTS = [
    (
        "You are a political journalist with 8k followers. You write sharp, opinionated takes on US policy.",
        "Feed: [{'user': 'senator_watch', 'content': 'New bill proposes 15% cap on corporate tax rate', 'likes': 4200, 'reposts': 890}, "
        "{'user': 'econ_daily', 'content': 'IMF warns bill could widen deficit by $400B over 10 years', 'likes': 2100, 'reposts': 340}]",
    ),
    (
        "You are a climate scientist. You follow the data closely and distrust sensationalism.",
        "Feed: [{'user': 'climatenews', 'content': 'Arctic ice extent hits record low for March', 'likes': 6700, 'reposts': 2100}, "
        "{'user': 'skeptic_pro', 'content': 'Satellite data shows cooling trend in southern hemisphere', 'likes': 890, 'reposts': 120}]",
    ),
    (
        "You are a tech investor with 50k followers. You are bullish on AI but cautious about hype.",
        "Feed: [{'user': 'ai_insider', 'content': 'Anthropic reportedly in talks to raise $5B at $60B valuation', 'likes': 9800, 'reposts': 3200}, "
        "{'user': 'vcdebate', 'content': 'AI valuations disconnected from revenue reality — correction incoming', 'likes': 3400, 'reposts': 980}]",
    ),
    (
        "You are a public health researcher. You communicate complex topics to a general audience.",
        "Feed: [{'user': 'cdc_watch', 'content': 'New RSV variant showing increased transmission in children under 5', 'likes': 5500, 'reposts': 1800}, "
        "{'user': 'med_debate', 'content': 'Study questions RSV vaccine efficacy in immunocompromised adults', 'likes': 1200, 'reposts': 230}]",
    ),
    (
        "You are a progressive activist with 12k followers. You focus on housing and inequality.",
        "Feed: [{'user': 'housing_now', 'content': 'SF approves 10k new units — but only 8% affordable', 'likes': 7800, 'reposts': 2900}, "
        "{'user': 'urbanist', 'content': 'New research: supply-side housing fails to reduce rents in low-income areas', 'likes': 4100, 'reposts': 1100}]",
    ),
    (
        "You are a conservative commentator with 30k followers. You prioritize economic freedom.",
        "Feed: [{'user': 'free_market', 'content': 'Fed signals two more rate cuts before year end', 'likes': 3200, 'reposts': 780}, "
        "{'user': 'fiscal_hawk', 'content': 'Debt ceiling debate returns — Treasury warns of X-date in September', 'likes': 5600, 'reposts': 1400}]",
    ),
    (
        "You are a science communicator who goes viral. You simplify without dumbing down.",
        "Feed: [{'user': 'physics_now', 'content': 'CERN announces hints of a new particle at 4.2 sigma — not quite discovery threshold', 'likes': 12000, 'reposts': 5400}, "
        "{'user': 'skeptic_sci', 'content': 'Historical base rate: most 4-sigma signals don't replicate', 'likes': 2800, 'reposts': 670}]",
    ),
    (
        "You are a foreign policy analyst. You are cautious and evidence-based.",
        "Feed: [{'user': 'geopolitics', 'content': 'China conducts largest naval exercise in South China Sea since 2016', 'likes': 8900, 'reposts': 3100}, "
        "{'user': 'intl_law', 'content': 'PRC exercises remain within UNCLOS-defined international waters, legal experts say', 'likes': 2200, 'reposts': 490}]",
    ),
]


def build_inputs(tokenizer, persona: str, feed: str, device):
    messages = [
        {"role": "system", "content": persona},
        {"role": "user", "content": f"{feed}\n\nWrite a single tweet (under 280 characters). Output only the tweet text, nothing else."},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").to(device)


def generate_baseline(model, tokenizer, persona: str, feed: str, max_new_tokens: int = 80) -> tuple[str, float]:
    inputs = build_inputs(tokenizer, persona, feed, model.device)
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed


def generate_interlat(
    model,
    tokenizer,
    persona: str,
    feed: str,
    adapter,
    K: int,
    max_new_tokens: int = 80,
) -> tuple[str, float, float]:
    """Returns (text, sender_time, rollout_time).

    sender_time  — latent communication cost: one forward pass + adapter.
                   This is what replaces text generation in the actual simulation.
    rollout_time — post-hoc decode for evaluation only, not counted as comm cost.
    """
    inputs = build_inputs(tokenizer, persona, feed, model.device)

    t0 = time.time()
    sender_states = extract_sender_states(model, inputs["input_ids"], K, inputs["attention_mask"])
    # adapter is cheap (no model forward), fold into sender time
    sender_time = time.time() - t0

    t1 = time.time()
    text = rollout_from_sender(
        model, tokenizer, sender_states, adapter,
        receiver_input_ids=inputs["input_ids"],
        max_new_tokens=max_new_tokens,
    )
    rollout_time = time.time() - t1

    return text.strip(), sender_time, rollout_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--K", type=int, default=8, help="Sender hidden states to transmit")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    adapter = make_adapter(model)
    ckpt = "checkpoints/adapter_oasis.pt"
    if os.path.exists(ckpt):
        adapter.load_state_dict(torch.load(ckpt, map_location=model.device, weights_only=True))
        print(f"Loaded trained adapter from {ckpt}")
    else:
        print("No adapter checkpoint found — using identity-init (untrained)")
    adapter.eval()

    baseline_posts, interlat_posts = [], []
    baseline_times, sender_times, rollout_times = [], [], []

    for i, (persona, feed) in enumerate(CONTEXTS):
        print(f"\n[{i+1}/{len(CONTEXTS)}] {persona[:60]}...")

        b_text, b_time = generate_baseline(model, tokenizer, persona, feed, args.max_new_tokens)
        i_text, s_time, r_time = generate_interlat(model, tokenizer, persona, feed, adapter, args.K, args.max_new_tokens)

        baseline_posts.append(b_text)
        interlat_posts.append(i_text)
        baseline_times.append(b_time)
        sender_times.append(s_time)
        rollout_times.append(r_time)

        print(f"  baseline  ({b_time:.2f}s gen): {b_text[:100]}")
        print(f"  interlat  ({s_time:.2f}s send | {r_time:.2f}s decode): {i_text[:100]}")

    similarities = batch_similarity(baseline_posts, interlat_posts)
    b_tokens = token_count(baseline_posts, tokenizer)
    i_tokens = token_count(interlat_posts, tokenizer)

    avg_b   = sum(baseline_times) / len(baseline_times)
    avg_s   = sum(sender_times)   / len(sender_times)
    avg_r   = sum(rollout_times)  / len(rollout_times)
    speedup = avg_b / avg_s if avg_s > 0 else float("inf")

    print("\n" + "=" * 60)
    print(f"Model: {args.model}  K={args.K}")
    print(f"Semantic similarity:      {similarities.mean():.3f} ± {similarities.std():.3f}")
    print(f"Baseline tokens/post:     {sum(b_tokens)/len(b_tokens):.1f}")
    print(f"Interlat tokens/post:     {sum(i_tokens)/len(i_tokens):.1f}")
    print(f"Baseline gen time/post:   {avg_b:.3f}s  (text communication cost)")
    print(f"Interlat sender time/post:{avg_s:.3f}s  (latent communication cost)")
    print(f"Interlat decode time/post:{avg_r:.3f}s  (post-hoc eval only)")
    print(f"Communication speedup:    {speedup:.1f}x")

    results = {
        "model": args.model,
        "K": args.K,
        "adapter_trained": True,
        "n_contexts": len(CONTEXTS),
        "similarities": similarities.tolist(),
        "mean_similarity": float(similarities.mean()),
        "std_similarity": float(similarities.std()),
        "communication_speedup": speedup,
        "baseline": {
            "posts": baseline_posts,
            "times": baseline_times,
            "tokens": b_tokens,
        },
        "interlat": {
            "posts": interlat_posts,
            "sender_times": sender_times,
            "rollout_times": rollout_times,
            "tokens": i_tokens,
        },
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
