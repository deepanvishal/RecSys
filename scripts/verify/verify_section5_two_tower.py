import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from utils.device import get_device
from config.config_loader import get_config
from models.two_tower.model import TwoTowerModel
from models.two_tower.train_two_tower import get_score_matrix, build_full_item_tensors
from evaluation.metrics import evaluate_vectorized


cfg = get_config()
device = get_device()
processed = Path('data/processed')
stats = json.load(open(processed / 'dataset_stats.json'))
n_users, n_items = stats['n_users'], stats['n_items']


model = TwoTowerModel(n_users, n_items, temperature=cfg['models']['two_tower']['temperature']).to(device)
model.load_state_dict(torch.load('artifacts/models/two_tower/two_tower_best.pt', map_location=device))


content_emb = np.load('artifacts/two_tower/item_content_emb.npy')
genre_matrix = np.load('artifacts/two_tower/item_genre_matrix.npy')
all_ids, all_content, all_genres = build_full_item_tensors(content_emb, genre_matrix, device)


val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
train_df = pd.read_parquet(processed / 'train.parquet')
user_meta = pd.read_parquet(processed / 'user_metadata.parquet')


score_matrix = get_score_matrix(
    model, all_ids, all_content, all_genres,
    user_meta, train_df, device,
    max_seq_len=cfg['models']['two_tower']['max_seq_len'],
)
metrics = evaluate_vectorized(score_matrix, val_df, train_df[['user_id', 'item_id']])


print('Two-Tower Best Model — full-catalog evaluation:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print(f'  CF baseline HR@10:  0.0830')
print(f'  Delta vs CF: {metrics["HR@10"] - 0.0830:+.4f}')
assert metrics['HR@10'] > 0.01, 'Two-Tower below noise floor'
print('Section 5 Two-Tower complete.')
