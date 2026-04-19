import torch
import torch.nn as nn


class SASRec(nn.Module):
    """
    SASRec with pre-norm transformer blocks. CE loss over full item set (SASRec+).
    Sequences are LEFT-padded with 0; index 0 of item_emb is the (zero) padding row.
    """
    def __init__(self, n_items, hidden=128, max_len=200,
                 num_blocks=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden = hidden
        self.max_len = max_len
        # +1 for padding index 0; real item ids stored at positions 1..n_items
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.ln_out = nn.LayerNorm(hidden)

    def forward(self, seq):
        # seq: (B, L) — item IDs, 0-padded on the LEFT
        B, L = seq.shape
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        # Causal mask (upper triangle = -inf). NOTE: no key_padding_mask — combining it
        # with causal mask makes early pad positions get all-masked rows -> NaN softmax.
        # Pad item embedding is zero (padding_idx=0) so it contributes ~nothing anyway.
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device) * float('-inf'), diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_out(x)
        return x  # (B, L, hidden)

    def predict(self, seq):
        """Embedding of the LAST position (most recent item)."""
        out = self.forward(seq)
        return out[:, -1, :]

    def score_all_items(self, seq):
        """(B, n_items) logits via dot product with item embedding table (skip pad row 0)."""
        user_emb = self.predict(seq)
        item_w = self.item_emb.weight[1:]  # (n_items, hidden)
        return user_emb @ item_w.T


class SASRecBlock(nn.Module):
    def __init__(self, hidden, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = x + residual
        residual = x
        x = self.ff(self.ln2(x)) + residual
        return x
