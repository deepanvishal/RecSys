"""
Two-Tower retrieval (genuinely new — CF and SVD use their own .retrieve() methods).
Also: precompute user + item embeddings via the trained Two-Tower towers,
and build/load a FAISS IndexFlatIP over the 128-dim item embeddings.
"""
import json
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
import torch
from utils.logger import get_logger
from utils.device import get_device


logger = get_logger('retrieval')


class FAISSRetriever:
    """FAISS IndexFlatIP on L2-normalized Two-Tower item embeddings (128-dim)."""

    def __init__(self, item_emb_path: str, index_path: str = None):
        item_emb = np.load(item_emb_path).astype(np.float32)
        faiss.normalize_L2(item_emb)
        self.item_emb = item_emb
        self.n_items, self.dim = item_emb.shape

        if index_path and Path(index_path).exists():
            self.index = faiss.read_index(str(index_path))
            logger.info(f'Loaded FAISS index from {index_path}')
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(item_emb)
            if index_path:
                Path(index_path).parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(index_path))
            logger.info(f'Built FAISS index: {self.n_items} items, dim={self.dim}')

    def retrieve(self, user_emb: np.ndarray, k: int = 100, exclude_ids: list = None) -> tuple:
        exclude = set(exclude_ids or [])
        fetch_k = k + len(exclude) + 10
        q = np.asarray(user_emb, dtype=np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, fetch_k)
        scores, ids = scores[0], ids[0]
        if exclude:
            mask = ~np.isin(ids, list(exclude))
            scores, ids = scores[mask], ids[mask]
        return ids[:k].astype(np.int32), scores[:k].astype(np.float32)


def _build_user_history_lookup(train_df, n_users, max_seq_len):
    """Two-Tower convention: unshifted item ids, left-padded with 0."""
    hist_arr = np.zeros((n_users, max_seq_len), dtype=np.int64)
    len_arr = np.ones(n_users, dtype=np.int64)
    grouped = (
        train_df.sort_values('timestamp')
        .groupby('user_id', sort=False)['item_id']
        .apply(lambda s: s.tolist()[-max_seq_len:])
    )
    for uid, items in grouped.items():
        if not items:
            continue
        hist_arr[uid, max_seq_len - len(items):] = items
        len_arr[uid] = len(items)
    return hist_arr, len_arr


def precompute_embeddings(cfg, device=None, force=False):
    """
    Precompute Two-Tower user + item embeddings (both 128-dim).
    Saves: artifacts/two_tower/user_emb.npy and item_emb.npy.
    """
    from models.two_tower.model import TwoTowerModel
    from models.two_tower.train_two_tower import build_full_item_tensors

    if device is None:
        device = get_device()
    processed = Path(cfg['data']['processed_dir'])
    artifacts = Path('artifacts')
    out_dir = artifacts / 'two_tower'
    out_dir.mkdir(parents=True, exist_ok=True)

    user_emb_path = out_dir / 'user_emb.npy'
    item_emb_path = out_dir / 'item_emb.npy'
    if user_emb_path.exists() and item_emb_path.exists() and not force:
        logger.info('user_emb.npy + item_emb.npy already exist; reuse.')
        return np.load(str(user_emb_path)), np.load(str(item_emb_path))

    stats = json.load(open(processed / 'dataset_stats.json'))
    n_users = stats['n_users']
    n_items = stats['n_items']
    tt_cfg = cfg['models']['two_tower']
    max_seq_len = tt_cfg['max_seq_len']

    model = TwoTowerModel(n_users, n_items, temperature=tt_cfg['temperature']).to(device)
    model.load_state_dict(torch.load(
        str(artifacts / 'models/two_tower/two_tower_best.pt'), map_location=device,
    ))
    model.eval()

    # ---- USER EMBEDDINGS ----
    train_df = pd.read_parquet(processed / 'train.parquet')
    user_meta = pd.read_parquet(processed / 'user_metadata.parquet')
    hist_arr, len_arr = _build_user_history_lookup(train_df, n_users, max_seq_len)
    demo = user_meta.set_index('user_id')[['gender_enc', 'age_enc', 'occupation']]
    demo_arr = demo.reindex(range(n_users)).fillna(0).values.astype(np.float32)

    batch_size = 512
    user_embs = np.zeros((n_users, 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            uids = torch.arange(start, end, dtype=torch.long, device=device)
            seqs = torch.tensor(hist_arr[start:end], dtype=torch.long, device=device)
            lens = torch.tensor(len_arr[start:end], dtype=torch.long)
            demos = torch.tensor(demo_arr[start:end], dtype=torch.float32, device=device)
            user_embs[start:end] = model.encode_users(uids, seqs, lens, demos).cpu().numpy()
    np.save(str(user_emb_path), user_embs)
    logger.info(f'Saved {user_emb_path}: {user_embs.shape}')

    # ---- ITEM EMBEDDINGS ----
    content_emb = np.load(str(out_dir / 'item_content_emb.npy'))
    genre_matrix = np.load(str(out_dir / 'item_genre_matrix.npy'))
    all_ids, all_content, all_genres = build_full_item_tensors(content_emb, genre_matrix, device)
    with torch.no_grad():
        item_embs = model.encode_items(all_ids, all_content, all_genres).cpu().numpy().astype(np.float32)
    np.save(str(item_emb_path), item_embs)
    logger.info(f'Saved {item_emb_path}: {item_embs.shape}')

    return user_embs, item_embs


def build_faiss_index(cfg=None):
    """Build/load FAISS index over the 128-dim Two-Tower item embeddings."""
    item_emb_path = 'artifacts/two_tower/item_emb.npy'
    if not Path(item_emb_path).exists():
        from config.config_loader import get_config
        precompute_embeddings(cfg if cfg is not None else get_config())
    return FAISSRetriever(item_emb_path, 'artifacts/two_tower/faiss.index')


if __name__ == '__main__':
    from config.config_loader import get_config
    cfg = get_config()
    precompute_embeddings(cfg)
    build_faiss_index(cfg)
    logger.info('retrieval.py complete')
