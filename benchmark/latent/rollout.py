import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from latent.adapter import InterlayAdapter


def rollout_from_sender(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sender_states: torch.Tensor,
    adapter: InterlayAdapter,
    receiver_input_ids: torch.Tensor,
    max_new_tokens: int = 128,
) -> str:
    """
    Interlat receiver: compress sender's K hidden states through the adapter
    into a prefix, prepend it to the receiver's own prompt, then autoregress.

    sender_states:      (1, K, d_h)
    receiver_input_ids: (1, seq_len)  — receiver's own prompt tokens
    """
    eos_id = tokenizer.eos_token_id
    device = model.device
    model_dtype = next(model.parameters()).dtype

    embed = model.model.embed_tokens if hasattr(model, "model") else model.transformer.wte
    embed_norm = embed.weight.norm(dim=-1).mean().item()

    bos_id = torch.tensor([[model.config.bos_token_id or model.config.eos_token_id]], device=device)
    query = embed(bos_id).float()  # (1, 1, d_h)

    with torch.no_grad():
        # Adapter prefix (1 token) from sender hidden states
        prefix = adapter(query, sender_states.float(), target_norm=embed_norm)
        prefix = prefix.to(model_dtype)                       # (1, 1, d_h)

        # Receiver's own prompt embeddings
        prompt_embeds = embed(receiver_input_ids).to(model_dtype)  # (1, seq_len, d_h)

        # [adapter_prefix | prompt_embeds] → full context for generation
        inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)  # (1, 1+seq_len, d_h)

        outputs = model(inputs_embeds=inputs_embeds, use_cache=True)

    past_kv = outputs.past_key_values
    cache_len = inputs_embeds.shape[1]

    first_token = outputs.logits[:, -1, :].argmax(dim=-1)
    if first_token.item() == eos_id:
        return ""

    generated = [first_token.item()]
    input_ids = first_token.unsqueeze(-1)
    position_ids = torch.tensor([[cache_len]], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_kv,
                position_ids=position_ids,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            if next_token.item() == eos_id:
                break
            generated.append(next_token.item())
            past_kv = outputs.past_key_values
            input_ids = next_token.unsqueeze(-1)
            position_ids = position_ids + 1

    return tokenizer.decode(generated, skip_special_tokens=True)
