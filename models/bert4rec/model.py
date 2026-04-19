import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """
    Bidirectional self-attention with Cloze masking. CE loss over full item set.
    Vocab: 0=pad, 1..n_items=items, n_items+1=MASK.
    Sequences are LEFT-padded with 0; inference appends MASK at the rightmost position.
    """
    def __init__(self, n_items, hidden=128, max_len=200,
                 num_blocks=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.hidden = hidden
        self.max_len = max_len
        self.mask_token = n_items + 1
        vocab_size = n_items + 2
        self.item_emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            BERT4RecBlock(hidden, num_heads, dropout) for _ in range(num_blocks)
        ])
        self.ln_out = nn.LayerNorm(hidden)

    def forward(self, seq):
        B, L = seq.shape
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        # No key_padding_mask: combined with bidirectional softmax it produces NaN at
        # pad-position queries (all keys masked). Pad item embedding is zero (padding_idx=0)
        # so contribution to non-pad outputs is small. Loss ignores pad positions anyway.
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return x  # (B, L, hidden)

    def score_all_items(self, seq):
        """
        Inference: seq has MASK token at last position.
        Returns (B, n_items) logits at the MASK position vs the n_items real item embeddings.
        """
        out = self.forward(seq)
        mask_out = out[:, -1, :]                              # (B, hidden) — MASK
        item_w = self.item_emb.weight[1:self.n_items + 1]     # (n_items, hidden)
        return mask_out @ item_w.T


class BERT4RecBlock(nn.Module):
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

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, need_weights=False)  # bidirectional, no causal mask
        x = x + residual
        residual = x
        x = self.ff(self.ln2(x)) + residual
        return x
