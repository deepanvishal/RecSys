import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from utils.device import get_device
from config.config_loader import get_config
from models.bert4rec.model import BERT4Rec
from models.bert4rec.dataset import build_user_sequences_bert
from models.bert4rec.train_bert4rec import get_score_matrix
from evaluation.metrics import evaluate_vectorized


cfg = get_config()
device = get_device()
processed = Path('data/processed')
stats = json.load(open(processed / 'dataset_stats.json'))
n_users, n_items = stats['n_users'], stats['n_items']

br_cfg = cfg['models']['bert4rec']
model = BERT4Rec(
    n_items=n_items,
    hidden=br_cfg['hidden'],
    max_len=br_cfg['max_len'],
    num_blocks=br_cfg['num_blocks'],
    num_heads=br_cfg['num_heads'],
)
model.load_state_dict(torch.load('artifacts/models/bert4rec/bert4rec_best.pt', map_location=device))
model = model.to(device)


train_df = pd.read_parquet(processed / 'train.parquet')
val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
user_seqs = build_user_sequences_bert(train_df, n_items, max_len=br_cfg['max_len'])


score_matrix = get_score_matrix(model, user_seqs, n_users, n_items, device)
metrics = evaluate_vectorized(score_matrix, val_df, train_df[['user_id', 'item_id']])


print('BERT4Rec Best Model — full-catalog evaluation:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print(f'  SASRec HR@10:       0.3328')
print(f'  Delta vs SASRec:   {metrics["HR@10"] - 0.3328:+.4f}')
print(f'  SVD HR@10:          0.1020')
assert metrics['HR@10'] > 0.01
print('Section 7 BERT4Rec complete.')
