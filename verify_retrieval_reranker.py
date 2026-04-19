import json
from pathlib import Path
import numpy as np
import pandas as pd
from models.cf.item_item_cf import ItemItemCF
from models.svd.als_model import ALSModel
from inference.retrieval import FAISSRetriever
from inference.reranker import Reranker
from inference.reranker_data import _build_lookups, build_features


processed = Path('data/processed')
train_df = pd.read_parquet(processed / 'train.parquet')
user_meta = pd.read_parquet(processed / 'user_metadata.parquet')
item_meta = pd.read_parquet(processed / 'item_metadata.parquet')
stats = json.load(open(processed / 'dataset_stats.json'))
n_items = stats['n_items']
user_emb = np.load('artifacts/two_tower/user_emb.npy')

user_history, n_history, item_genre, user_genre, user_demo, item_pop = _build_lookups(
    train_df, user_meta, item_meta, n_items,
)

# Pick a warm user with the most history
uid = int(train_df.groupby('user_id').size().idxmax())
history = user_history[uid]
print(f'Test user: {uid}, history length: {len(history)}')


# CF retrieve via extended class
cf = ItemItemCF.load('artifacts/models/cf/item_item_best.pkl')
cf_ids, cf_scores = cf.retrieve(history, k=100, exclude_ids=history)
assert len(cf_ids) > 0 and len(cf_ids) <= 100
print(f'ItemItemCF.retrieve OK: {len(cf_ids)} candidates, top score={cf_scores[0]:.4f}')


# SVD retrieve via extended class
svd = ALSModel.load('artifacts/models/svd/als_best.pkl')
svd_ids, svd_scores = svd.retrieve(history, k=100, exclude_ids=history)
assert len(svd_ids) > 0
print(f'ALSModel.retrieve OK: {len(svd_ids)} candidates, top score={svd_scores[0]:.4f}')


# FAISS retrieve via FAISSRetriever
faiss_r = FAISSRetriever(
    'artifacts/two_tower/item_emb.npy',
    'artifacts/two_tower/faiss.index',
)
tt_ids, tt_scores = faiss_r.retrieve(user_emb[uid], k=100, exclude_ids=history)
assert len(tt_ids) > 0
print(f'FAISSRetriever.retrieve OK: {len(tt_ids)} candidates, dim={faiss_r.dim}, top score={tt_scores[0]:.4f}')


# Reranker
feat_df = build_features(uid, cf_ids[:50], cf_scores[:50],
                          user_demo, n_history, user_genre, item_genre, item_pop)
reranker = Reranker()
reranked = reranker.rerank(feat_df, k=10)
assert len(reranked) == 10
assert not any(i in history for i in reranked)
print(f'Reranker OK: top-10 = {reranked}')
print('All checks passed.')
