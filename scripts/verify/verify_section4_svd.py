import pandas as pd
from models.svd.als_model import ALSModel
from evaluation.metrics import evaluate_model


model = ALSModel.load('artifacts/models/svd/als_best.pkl')
val_df = pd.read_parquet('data/processed/val.parquet')[['user_id', 'item_id']]


metrics = evaluate_model(model.recommend, val_df)
print('Best ALS Model — full-catalog evaluation:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print(f'  CF baseline HR@10: 0.0830')
print(f'  Delta vs CF: {metrics["HR@10"] - 0.0830:+.4f}')


train = pd.read_parquet('data/processed/train.parquet')
sample_user = train.groupby('user_id').head(3).groupby('user_id').first().reset_index()
uid = int(sample_user.iloc[0]['user_id'])
items = train[train['user_id'] == uid].head(3)['item_id'].tolist()
emb = model.fold_in(items)
recs = model.recommend_from_embedding(emb, items, N=10)
print(f'\nFold-in test: user {uid}, history={items}')
print(f'  Recommendations: {recs}')
assert len(recs) == 10, 'Expected 10 recs from fold-in'
print('OK: fold-in working')


assert metrics['HR@10'] > 0.01
print(f'\nSection 4 SVD complete. HR@10={metrics["HR@10"]:.4f}')
