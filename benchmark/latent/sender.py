import torch


def extract_sender_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    K: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Interlat sender: run one forward pass, return last K hidden states.
    Shape: (batch, K, d_h)
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    last_layer = outputs.hidden_states[-1]           # (batch, seq_len, d_h)
    return last_layer[:, -K:, :].contiguous()        # (batch, K, d_h)
