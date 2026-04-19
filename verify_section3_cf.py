import scipy.sparse as sp
import pandas as pd
from pathlib import Path
from models.cf.item_item_cf import ItemItemCF
from evaluation.metrics import evaluate_model


matrix = sp.load_npz('artifacts/cache/cf_interaction_matrix.npz')
val_df = pd.read_parquet('data/processed/val.parquet')[['user_id', 'item_id']]
model = ItemItemCF.load('artifacts/models/cf/item_item_best.pkl')


metrics = evaluate_model(model.recommend, val_df)
print('ItemItemCF Best Model Metrics:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')


assert metrics['HR@10'] > 0.01, f'HR@10 too low: {metrics["HR@10"]}'
print(f'\nSection 3 CF complete. HR@10={metrics["HR@10"]:.4f}')
