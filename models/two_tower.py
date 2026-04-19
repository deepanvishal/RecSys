import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('two_tower')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UserTower(nn.Module):
    """
    User tower: mean of interacted item ST embeddings -> MLP -> 128-dim L2-normalized output.
    No user ID table — handles new users naturally via mean pooling.
    Zero interactions -> zero vector -> mapped to generic embedding (trending proxy).
    """
    def __init__(self, item_emb_dim=384, hidden_dims=[256, 128], output_dim=128, dropout=0.2):
        super().__init__()
        self.mlp = MLP(item_emb_dim, hidden_dims, output_dim, dropout)

    def forward(self, user_history_emb):
        out = self.mlp(user_history_emb)
        return F.normalize(out, dim=-1)


class ItemTower(nn.Module):
    """
    Item tower: ST embedding (384) + price_bucket one-hot (6) + popularity_rank (1) -> MLP -> 128-dim.
    Content-driven — new items get meaningful vectors immediately.
    price_bucket: 5 quantiles (0-4) + -1 for unknown = 6 classes total.
    """
    def __init__(self, item_emb_dim=384, n_price_buckets=6, hidden_dims=[256, 128], output_dim=128, dropout=0.2):
        super().__init__()
        input_dim = item_emb_dim + n_price_buckets + 1
        self.n_price_buckets = n_price_buckets
        self.mlp = MLP(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, item_emb, price_bucket, popularity_rank):
        # Shift -1 to index 0, 0-4 to indices 1-5
        price_idx = (price_bucket + 1).clamp(0, self.n_price_buckets - 1)
        price_onehot = F.one_hot(price_idx, num_classes=self.n_price_buckets).float()
        pop = popularity_rank.unsqueeze(-1)
        x = torch.cat([item_emb, price_onehot, pop], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Full two-tower model.
    Loss: InfoNCE with in-batch negatives.
    Inference: encode items once -> FAISS index. Encode user on-the-fly -> ANN search.
    """
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = get_config()
        tt_cfg = cfg['models']['two_tower']
        self.temperature = tt_cfg['temperature']
        self.output_dim = tt_cfg['item_embedding_dim']
        self.user_tower = UserTower(
            item_emb_dim=384,
            hidden_dims=tt_cfg['mlp_hidden_dims'],
            output_dim=self.output_dim,
            dropout=tt_cfg['dropout'],
        )
        self.item_tower = ItemTower(
            item_emb_dim=384,
            n_price_buckets=6,
            hidden_dims=tt_cfg['mlp_hidden_dims'],
            output_dim=self.output_dim,
            dropout=tt_cfg['dropout'],
        )

    def forward(self, user_history_emb, item_emb, price_bucket, popularity_rank):
        user_vec = self.user_tower(user_history_emb)
        item_vec = self.item_tower(item_emb, price_bucket, popularity_rank)
        return user_vec, item_vec

    def infonce_loss(self, user_vec, item_vec):
        """InfoNCE with in-batch negatives. Diagonal = positives."""
        logits = (user_vec @ item_vec.T) / self.temperature
        labels = torch.arange(len(user_vec), device=user_vec.device)
        return F.cross_entropy(logits, labels)

    def encode_user(self, user_history_emb):
        self.eval()
        with torch.no_grad():
            return self.user_tower(user_history_emb)

    def encode_items(self, item_emb, price_bucket, popularity_rank):
        self.eval()
        with torch.no_grad():
            return self.item_tower(item_emb, price_bucket, popularity_rank)
