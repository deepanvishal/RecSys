import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, n_items, id_dim=64, content_dim=384, genre_dim=18, out_dim=128):
        super().__init__()
        self.id_emb = nn.Embedding(n_items, id_dim, padding_idx=0)
        self.content_proj = nn.Linear(content_dim, 128)
        self.genre_proj = nn.Linear(genre_dim, 32)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + 128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, item_ids, content_embs, genre_vecs):
        id_feat = self.id_emb(item_ids)                          # (B, 64)
        content_feat = F.relu(self.content_proj(content_embs))   # (B, 128)
        genre_feat = F.relu(self.genre_proj(genre_vecs))         # (B, 32)
        x = torch.cat([id_feat, content_feat, genre_feat], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, n_users, n_items, id_dim=64, item_emb_dim=128,
                 demo_in=3, demo_dim=32, out_dim=128, max_seq_len=200):
        super().__init__()
        self.user_id_emb = nn.Embedding(n_users, id_dim)
        self.hist_item_emb = nn.Embedding(n_items, item_emb_dim, padding_idx=0)
        self.gru = nn.GRU(item_emb_dim, item_emb_dim, batch_first=True)
        self.demo_proj = nn.Linear(demo_in, demo_dim)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + item_emb_dim + demo_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, user_ids, hist_items, hist_lens, demo_feats):
        uid_feat = self.user_id_emb(user_ids)           # (B, 64)
        hist_embs = self.hist_item_emb(hist_items)      # (B, seq, 128)
        packed = nn.utils.rnn.pack_padded_sequence(
            hist_embs, hist_lens.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.gru(packed)                       # (1, B, 128)
        hist_feat = h_n.squeeze(0)                      # (B, 128)
        demo_feat = F.relu(self.demo_proj(demo_feats.float()))
        x = torch.cat([uid_feat, hist_feat, demo_feat], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.item_tower = ItemTower(n_items)
        self.user_tower = UserTower(n_users, n_items)

    def encode_items(self, item_ids, content_embs, genre_vecs):
        return self.item_tower(item_ids, content_embs, genre_vecs)

    def encode_users(self, user_ids, hist_items, hist_lens, demo_feats):
        return self.user_tower(user_ids, hist_items, hist_lens, demo_feats)

    def forward(self, user_ids, hist_items, hist_lens, demo_feats,
                item_ids, content_embs, genre_vecs):
        u_emb = self.encode_users(user_ids, hist_items, hist_lens, demo_feats)
        i_emb = self.encode_items(item_ids, content_embs, genre_vecs)
        logits = (u_emb @ i_emb.T) / self.temperature
        return logits, u_emb, i_emb
