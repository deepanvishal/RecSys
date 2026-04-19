"""
Precompute Two-Tower user AND item embeddings (both 128-dim, learned output space).
Saves:
  artifacts/two_tower/user_emb.npy   (n_users, 128)
  artifacts/two_tower/item_emb.npy   (n_items, 128)  -- this is what FAISS indexes
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config.config_loader import get_config
from utils.logger import get_logger
from utils.device import get_device
from models.two_tower.model import TwoTowerModel
from models.two_tower.train_two_tower import build_full_item_tensors


logger = get_logger('precompute_embeddings')


def _build_user_history_lookup(train_df, n_users, max_seq_len):
    """Same convention as train_two_tower (unshifted item ids, left-pad with 0)."""
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


def precompute():
    cfg = get_config()
    device = get_device()
    processed = Path(cfg['data']['processed_dir'])
    artifacts = Path('artifacts')
    out_dir = artifacts / 'two_tower'
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = json.load(open(processed / 'dataset_stats.json'))
    n_users = stats['n_users']
    n_items = stats['n_items']
    tt_cfg = cfg['models']['two_tower']
    max_seq_len = tt_cfg['max_seq_len']

    logger.info('Loading Two-Tower model...')
    model = TwoTowerModel(n_users, n_items, temperature=tt_cfg['temperature']).to(device)
    model.load_state_dict(torch.load(
        str(artifacts / 'models/two_tower/two_tower_best.pt'), map_location=device,
    ))
    model.eval()

    train_df = pd.read_parquet(processed / 'train.parquet')
    user_meta = pd.read_parquet(processed / 'user_metadata.parquet')

    # ---- USER EMBEDDINGS ----
    logger.info(f'Encoding {n_users} users via user_tower...')
    hist_arr, len_arr = _build_user_history_lookup(train_df, n_users, max_seq_len)
    demo = user_meta.set_index('user_id')[['gender_enc', 'age_enc', 'occupation']]
    demo_arr = demo.reindex(range(n_users)).fillna(0).values.astype(np.float32)

    batch_size = 512
    user_embs = np.zeros((n_users, tt_cfg['item_embedding_dim'] if 'item_embedding_dim' in tt_cfg else 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            uids = torch.arange(start, end, dtype=torch.long, device=device)
            seqs = torch.tensor(hist_arr[start:end], dtype=torch.long, device=device)
            lens = torch.tensor(len_arr[start:end], dtype=torch.long)
            demos = torch.tensor(demo_arr[start:end], dtype=torch.float32, device=device)
            emb = model.encode_users(uids, seqs, lens, demos)
            user_embs[start:end] = emb.cpu().numpy()

    np.save(str(out_dir / 'user_emb.npy'), user_embs)
    logger.info(f'Saved user embeddings: {user_embs.shape} -> {out_dir}/user_emb.npy')

    # ---- ITEM EMBEDDINGS ----
    logger.info(f'Encoding {n_items} items via item_tower...')
    content_emb = np.load(str(out_dir / 'item_content_emb.npy'))
    genre_matrix = np.load(str(out_dir / 'item_genre_matrix.npy'))
    all_ids, all_content, all_genres = build_full_item_tensors(content_emb, genre_matrix, device)
    with torch.no_grad():
        item_embs = model.encode_items(all_ids, all_content, all_genres).cpu().numpy().astype(np.float32)
    np.save(str(out_dir / 'item_emb.npy'), item_embs)
    logger.info(f'Saved item embeddings: {item_embs.shape} -> {out_dir}/item_emb.npy')

    return user_embs, item_embs


if __name__ == '__main__':
    precompute()
