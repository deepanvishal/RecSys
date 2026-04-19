from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import numpy as np


checks = []


# Check output files exist
required_files = [
    'data/raw/electronics.parquet',
    'data/processed/train.parquet',
    'data/processed/val.parquet',
    'data/processed/test.parquet',
    'data/processed/full.parquet',
    'data/processed/item_metadata.parquet',
    'data/processed/user_map.json',
    'data/processed/item_map.json',
    'data/processed/cohorts/user_cohorts.parquet',
    'data/processed/simulation/daily_snapshots.json',
    'artifacts/cache/cf_interaction_matrix.npz',
    'artifacts/cache/svd_item_user_matrix.npz',
    'artifacts/embeddings/item_title_embeddings.npy',
]
for f in required_files:
    exists = Path(f).exists()
    checks.append((f, exists))
    print(f"{'OK' if exists else 'MISSING'}: {f}")


# Check data contracts
train = pd.read_parquet('data/processed/train.parquet')
required_cols = ['user_id', 'item_id', 'rating', 'implicit_feedback',
                 'timestamp', 'user_history_len', 'price_bucket', 'popularity_rank']
for col in required_cols:
    ok = col in train.columns
    checks.append((f'col:{col}', ok))
    print(f"{'OK' if ok else 'MISSING COL'}: {col}")


# Check no data leakage
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')
no_leak = train['timestamp'].max() <= val['timestamp'].min()
checks.append(('no_temporal_leakage', no_leak))
print(f"{'OK' if no_leak else 'LEAKAGE DETECTED'}: temporal split integrity")


# Check cohorts
cohorts = pd.read_parquet('data/processed/cohorts/user_cohorts.parquet')
for c in ['cold', 'sparse_low', 'sparse_high', 'warm']:
    n = len(cohorts[cohorts['cohort'] == c])
    print(f'Cohort {c}: {n} users')


failed = [f for f, ok in checks if not ok]
if failed:
    print(f'FAILED: {failed}')
else:
    print('Section 2 complete. All checks passed. Proceed to Section 3.')
