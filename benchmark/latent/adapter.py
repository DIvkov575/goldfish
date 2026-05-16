import torch
import torch.nn as nn


class InterlayAdapter(nn.Module):
    """
    Interlat receiver adapter.
    Takes K sender hidden states and produces a prefix the receiver can condition on.

    Architecture (from paper §3.2):
      cross-attention (query=receiver context, key/value=sender states)
      → LayerNorm
      → Linear projection back to d_h

    Training: minimise reconstruction loss of receiver output vs. text-decoded ground truth.
    Untrained (identity init): returns mean-pooled sender states — meaningful upper bound.
    """

    def __init__(self, d_h: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_h, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_h)
        self.proj = nn.Linear(d_h, d_h)
        self._init_identity()

    def _init_identity(self):
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        sender_states: torch.Tensor,
        target_norm: float | None = None,
    ) -> torch.Tensor:
        """
        query:         (batch, q_len, d_h)  — e.g. receiver's BOS embedding
        sender_states: (batch, K, d_h)
        target_norm:   if set, rescale output to this L2 norm (match embed table)
        returns:       (batch, q_len, d_h)  — prefix to prepend to receiver context
        """
        attn_out, _ = self.cross_attn(query, sender_states, sender_states)
        out = self.proj(self.norm(attn_out))
        if target_norm is not None:
            current = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out = out / current * target_norm
        return out


def make_adapter(model: torch.nn.Module, num_heads: int = 8) -> InterlayAdapter:
    d_h = model.config.hidden_size
    adapter = InterlayAdapter(d_h, num_heads)
    # Keep adapter in float32: float16 gradient overflow causes NaN during training
    return adapter.to(model.device)
