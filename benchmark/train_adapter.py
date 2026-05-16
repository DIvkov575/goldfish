"""
Train the Interlat receiver adapter on OASIS agent post-generation data.

Only the adapter weights are trained; the base model is frozen.

Loss (from §3.3 of arXiv 2511.09149):
  L = L_lm + λ * L_contrast

  L_lm       — cross-entropy on the target post tokens given the adapter prefix
  L_contrast — JS-divergence margin loss: pushes real-prefix distribution away
               from random (mismatched) prefix distribution

Usage:
    python train_adapter.py \
        --data data/oasis_interlat_train.json \
        --model Qwen/Qwen3-4B \
        --epochs 10 \
        --lr 1e-4 \
        --output checkpoints/adapter_oasis.pt
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent.adapter import InterlayAdapter, make_adapter


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OASISDataset(Dataset):
    def __init__(self, records, tokenizer, max_target_len: int = 80):
        self.records = records
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        hidden = torch.load(rec["hidden_state"], weights_only=True)["hidden"]  # (K, d_h)
        target = rec["conversations"][1]["value"]
        target_ids = self.tokenizer.encode(
            target,
            add_special_tokens=False,
            max_length=self.max_target_len,
            truncation=True,
        )
        return {"hidden": hidden, "target_ids": torch.tensor(target_ids, dtype=torch.long)}


def collate(batch, pad_id: int):
    hidden = torch.stack([b["hidden"] for b in batch])
    max_len = max(b["target_ids"].shape[0] for b in batch)
    target_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["target_ids"].shape[0]
        target_ids[i, :L] = b["target_ids"]
    return {"hidden": hidden, "target_ids": target_ids}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def training_step(
    model,
    adapter: InterlayAdapter,
    embed,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    contrast_weight: float,
    contrast_margin: float,
    embed_norm: float = 1.0,
):
    """
    hidden:     (B, K, d_h)
    target_ids: (B, T)  padded with pad_id

    Returns total_loss, lm_loss_val, js_val
    """
    B = hidden.shape[0]
    device = hidden.device
    model_dtype = next(model.parameters()).dtype

    # --- query: BOS embedding (float32 for adapter) ---
    bos_id = torch.tensor(
        [[model.config.bos_token_id or model.config.eos_token_id]],
        device=device,
    ).expand(B, 1)
    query = embed(bos_id).float()                      # (B, 1, d_h) float32

    # --- adapter prefix (float32 adapter, rescaled to match embed norm) ---
    prefix_f32 = adapter(query, hidden.float(), target_norm=embed_norm)  # (B, 1, d_h) float32
    prefix = prefix_f32.to(model_dtype)               # cast to model dtype for forward

    # --- target token embeddings ---
    target_embeds = embed(target_ids).to(model_dtype)  # (B, T, d_h)

    # inputs_embeds = [prefix, e(t0) .. e(tT-2)]  → predict t0 .. tT-1
    inputs_embeds = torch.cat([prefix, target_embeds[:, :-1, :]], dim=1)  # (B, T, d_h)

    # labels: -100 for prefix slot, then the target ids
    labels = torch.cat([
        torch.full((B, 1), -100, dtype=torch.long, device=device),
        target_ids[:, :-1],
    ], dim=1)                                          # (B, T)

    # attention mask: 1 everywhere except padding in target
    pad_mask = (target_ids != pad_id)                  # (B, T)
    attn_mask = torch.cat([
        torch.ones(B, 1, dtype=torch.bool, device=device),
        pad_mask[:, :-1],
    ], dim=1)                                          # (B, T)

    # --- LM forward ---
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
    logits_f32 = out.logits.float()                   # (B, T, vocab) in float32

    lm_loss = F.cross_entropy(
        logits_f32.reshape(-1, logits_f32.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )

    # --- random contrast forward (no grad through model) ---
    shuffled = hidden[torch.randperm(B, device=device)]
    rand_prefix = adapter(query, shuffled.float(), target_norm=embed_norm).to(model_dtype)
    rand_embeds = torch.cat([rand_prefix, target_embeds[:, :-1, :]], dim=1)

    with torch.no_grad():
        rand_out = model(inputs_embeds=rand_embeds, attention_mask=attn_mask, use_cache=False)
    rand_logits_f32 = rand_out.logits.float()

    # JS divergence in float32 over valid (non-pad) positions
    valid = attn_mask.reshape(-1)
    p = F.softmax(logits_f32.reshape(-1, logits_f32.shape[-1])[valid], dim=-1)
    q = F.softmax(rand_logits_f32.reshape(-1, rand_logits_f32.shape[-1])[valid], dim=-1).detach()
    m = (0.5 * (p + q)).clamp(min=1e-8)
    js = 0.5 * (
        F.kl_div(m.log(), p, reduction="batchmean")
        + F.kl_div(m.log(), q, reduction="batchmean")
    )
    contrast_loss = torch.clamp(contrast_margin - js, min=0.0)

    total = lm_loss + contrast_weight * contrast_loss
    return total, lm_loss.item(), js.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/oasis_interlat_train.json")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-target-len", type=int, default=80)
    parser.add_argument("--contrast-weight", type=float, default=2.0)
    parser.add_argument("--contrast-margin", type=float, default=0.69)
    parser.add_argument("--output", default="checkpoints/adapter_oasis.pt")
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    embed = (model.model.embed_tokens
             if hasattr(model, "model") else model.transformer.wte)

    with torch.no_grad():
        embed_norm = embed.weight.norm(dim=-1).mean().item()
    print(f"Target embedding norm: {embed_norm:.4f}")

    adapter = make_adapter(model)
    adapter.train()
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    with open(args.data) as f:
        splits = json.load(f)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    train_ds = OASISDataset(splits["train"], tokenizer, args.max_target_len)
    eval_ds  = OASISDataset(splits["eval"],  tokenizer, args.max_target_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
    )

    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dl))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_eval_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss = lm_total = 0.0

        for batch in train_dl:
            hidden = batch["hidden"].to(device)
            target_ids = batch["target_ids"].to(device)

            optimizer.zero_grad()
            loss, lm_val, js_val = training_step(
                model, adapter, embed,
                hidden, target_ids, pad_id,
                args.contrast_weight, args.contrast_margin,
                embed_norm=embed_norm,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            lm_total   += lm_val

        avg_loss = total_loss / len(train_dl)
        avg_lm   = lm_total  / len(train_dl)

        if epoch % args.eval_every == 0:
            adapter.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in eval_dl:
                    hidden = batch["hidden"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    loss, _, _ = training_step(
                        model, adapter, embed,
                        hidden, target_ids, pad_id,
                        args.contrast_weight, args.contrast_margin,
                        embed_norm=embed_norm,
                    )
                    eval_loss += loss.item()
            eval_loss /= len(eval_dl)

            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train={avg_loss:.4f} (lm={avg_lm:.4f})  "
                  f"eval={eval_loss:.4f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(adapter.state_dict(), out_path)
                print(f"  -> saved best adapter ({out_path})")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_loss:.4f} (lm={avg_lm:.4f})")

    print(f"\nTraining complete. Best eval loss: {best_eval_loss:.4f}")
    print(f"Adapter saved to {out_path}")


if __name__ == "__main__":
    main()
