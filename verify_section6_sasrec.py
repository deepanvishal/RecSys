import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from utils.device import get_device
from config.config_loader import get_config
from models.sasrec.model import SASRec
from models.sasrec.dataset import build_user_sequences
from models.sasrec.train_sasrec import get_score_matrix
from evaluation.metrics import evaluate_vectorized


cfg = get_config()
device = get_device()
processed = Path('data/processed')
stats = json.load(open(processed / 'dataset_stats.json'))
n_users, n_items = stats['n_users'], stats['n_items']

sr_cfg = cfg['models']['sasrec']
model = SASRec(
    n_items=n_items,
    hidden=sr_cfg['hidden'],
    max_len=sr_cfg['max_len'],
    num_blocks=sr_cfg['num_blocks'],
    num_heads=sr_cfg['num_heads'],
)
model.load_state_dict(torch.load('artifacts/models/sasrec/sasrec_best.pt', map_location=device))
model = model.to(device)


train_df = pd.read_parquet(processed / 'train.parquet')
val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
user_seqs = build_user_sequences(train_df, max_len=sr_cfg['max_len'])


score_matrix = get_score_matrix(model, user_seqs, n_users, n_items, device)
metrics = evaluate_vectorized(score_matrix, val_df, train_df[['user_id', 'item_id']])


print('SASRec Best Model — full-catalog evaluation:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print(f'  SVD baseline HR@10: 0.1020')
print(f'  CF  baseline HR@10: 0.0830')
print(f'  Delta vs SVD: {metrics["HR@10"] - 0.1020:+.4f}')
assert metrics['HR@10'] > 0.01
print('Section 6 SASRec complete.')
