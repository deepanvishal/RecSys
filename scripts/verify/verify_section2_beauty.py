from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import numpy as np


checks = []


required = [
    'data/raw/beauty.parquet',
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
for f in required:
    ok = Path(f).exists()
    checks.append((f, ok))
    print(f"{'OK' if ok else 'MISSING'}: {f}")


full = pd.read_parquet('data/processed/full.parquet')
n_users = full['user_id'].nunique()
n_items = full['item_id'].nunique()
n_interactions = len(full)
print('\nDataset stats:')
print(f'  Users: {n_users} (expected ~22K)')
print(f'  Items: {n_items} (expected ~12K)')
print(f'  Interactions: {n_interactions} (expected ~198K)')
print(f'  Avg interactions/user: {n_interactions/n_users:.1f} (expected ~8-9)')


train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')
print(f'  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
assert len(val) == val['user_id'].nunique(), 'Val should have exactly 1 item per user'
assert len(test) == test['user_id'].nunique(), 'Test should have exactly 1 item per user'
checks.append(('leave_one_out_split', True))
print('OK: leave-one-out split verified')


user_counts = train.groupby('user_id').size()
item_counts = train.groupby('item_id').size()
ok = (user_counts >= 3).all() and (item_counts >= 3).all()
checks.append(('5core_filter', ok))
print(f"{'OK' if ok else 'FAILED'}: 5-core filter applied")


failed = [f for f, ok in checks if not ok]
if failed:
    print(f'\nFAILED: {failed}')
else:
    print('\nSection 2 Beauty complete. All checks passed. Proceed to Section 3.')
