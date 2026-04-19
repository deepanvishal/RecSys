from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import numpy as np
import json


required = [
    'data/raw/ml-1m/ratings.dat',
    'data/raw/ml-1m/movies.dat',
    'data/raw/ml-1m/users.dat',
    'data/processed/train.parquet',
    'data/processed/val.parquet',
    'data/processed/test.parquet',
    'data/processed/full.parquet',
    'data/processed/item_metadata.parquet',
    'data/processed/user_metadata.parquet',
    'data/processed/dataset_stats.json',
    'data/processed/cohorts/user_cohorts.parquet',
    'data/processed/simulation/daily_snapshots.json',
    'artifacts/cache/cf_interaction_matrix.npz',
    'artifacts/cache/svd_item_user_matrix.npz',
    'artifacts/embeddings/item_title_embeddings.npy',
]
failed = []
for f in required:
    ok = Path(f).exists()
    print(f"{'OK' if ok else 'MISSING'}: {f}")
    if not ok:
        failed.append(f)


full = pd.read_parquet('data/processed/full.parquet')
train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')
print('\nStats:')
print(f'  Users:         {full.user_id.nunique()} (expected 6040)')
print(f'  Items:         {full.item_id.nunique()} (expected ~3700-3900)')
print(f'  Total ratings: {len(full)} (expected ~1,000,209)')
print(f'  Avg/user:      {len(full)/full.user_id.nunique():.1f} (expected ~165)')
print(f'  Train:         {len(train)}')
print(f'  Val:           {len(val)} (expected = n_users)')
print(f'  Test:          {len(test)} (expected = n_users)')


assert len(val) == val['user_id'].nunique(), 'Val must be 1 per user'
assert len(test) == test['user_id'].nunique(), 'Test must be 1 per user'
print('OK: leave-one-out split verified')


um = pd.read_parquet('data/processed/user_metadata.parquet')
assert 'gender_enc' in um.columns, 'Missing gender_enc'
assert 'age_enc' in um.columns, 'Missing age_enc'
assert 'occupation' in um.columns, 'Missing occupation'
print('OK: user demographics verified')


emb = np.load('artifacts/embeddings/item_title_embeddings.npy')
print(f'  ST embeddings: {emb.shape}  ({emb.nbytes/1e6:.1f} MB)')


if failed:
    print(f'\nFAILED: {failed}')
else:
    print('\nSection 2 ML-1M complete. Proceed to Section 3 (CF).')
