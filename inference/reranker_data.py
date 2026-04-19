"""
Build LightGBM reranker training data per retriever.
Uses extended model.retrieve() methods (no wrapper classes for CF/SVD).
Two-Tower retrieval via FAISSRetriever.

Perf: precomputes user_history dict, n_history dict, user_genre_profile dict,
and item_genre matrix ONCE — avoids per-user/per-candidate DataFrame scans.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.cf.item_item_cf import ItemItemCF
from models.svd.als_model import ALSModel
from inference.retrieval import FAISSRetriever


logger = get_logger('reranker_data')


GENRES = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']


def _build_lookups(train_df, user_meta, item_meta, n_items):
    """Precompute all per-user / per-item dicts in one pass."""
    user_history = (
        train_df.sort_values('timestamp')
        .groupby('user_id', sort=False)['item_id']
        .apply(list).to_dict()
    )
    n_history = {uid: len(items) for uid, items in user_history.items()}

    item_genre = np.zeros((n_items, len(GENRES)), dtype=np.float32)
    for _, row in item_meta.iterrows():
        iid = int(row['item_id'])
        if iid < n_items:
            item_genre[iid] = row['genre_vector']

    user_genre = {}
    for uid, items in user_history.items():
        valid = [i for i in items if i < n_items]
        if valid:
            user_genre[uid] = item_genre[valid].mean(axis=0)
        else:
            user_genre[uid] = np.zeros(len(GENRES), dtype=np.float32)

    user_demo = (
        user_meta.set_index('user_id')[['gender_enc', 'age_enc', 'occupation']].to_dict('index')
    )

    item_pop = train_df.groupby('item_id').size().to_dict()
    return user_history, n_history, item_genre, user_genre, user_demo, item_pop


def build_features(uid, candidate_ids, candidate_scores,
                   user_demo, n_history, user_genre, item_genre, item_pop):
    demo = user_demo.get(uid, {'gender_enc': 0, 'age_enc': 0, 'occupation': 0})
    nh = n_history.get(uid, 0)
    ug = user_genre.get(uid, np.zeros(len(GENRES), dtype=np.float32))
    cids = np.asarray(candidate_ids, dtype=np.int64)
    cscores = np.asarray(candidate_scores, dtype=np.float32)
    ig = item_genre[cids]                                       # (K, 18)
    affinity = (ig * ug).sum(axis=1).astype(np.float32)
    pops = np.array([item_pop.get(int(i), 0) for i in cids], dtype=np.float32)

    out = {
        'user_id': np.full(len(cids), int(uid), dtype=np.int64),
        'item_id': cids,
        'retrieval_score': cscores,
        'gender_enc': np.full(len(cids), int(demo['gender_enc']), dtype=np.int8),
        'age_enc': np.full(len(cids), int(demo['age_enc']), dtype=np.int8),
        'occupation': np.full(len(cids), int(demo['occupation']), dtype=np.int8),
        'n_history': np.full(len(cids), int(nh), dtype=np.int32),
        'genre_affinity': affinity,
        'item_popularity': pops,
    }
    for gi, g in enumerate(GENRES):
        out[f'user_genre_{g}'] = np.full(len(cids), float(ug[gi]), dtype=np.float32)
    return pd.DataFrame(out)


def build_reranker_dataset(model_name: str, top_k: int = 100):
    set_seed()
    cfg = get_config()
    processed = Path(cfg['data']['processed_dir'])
    out_dir = Path('artifacts/reranker')
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(processed / 'train.parquet')
    val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
    user_meta = pd.read_parquet(processed / 'user_metadata.parquet')
    item_meta = pd.read_parquet(processed / 'item_metadata.parquet')
    stats = json.load(open(processed / 'dataset_stats.json'))
    n_items = stats['n_items']

    user_history, n_history, item_genre, user_genre, user_demo, item_pop = _build_lookups(
        train_df, user_meta, item_meta, n_items,
    )

    if model_name == 'cf':
        model = ItemItemCF.load('artifacts/models/cf/item_item_best.pkl')
        retrieve_fn = lambda uid, hist: model.retrieve(hist, k=top_k, exclude_ids=hist)
    elif model_name == 'svd':
        model = ALSModel.load('artifacts/models/svd/als_best.pkl')
        retrieve_fn = lambda uid, hist: model.retrieve(hist, k=top_k, exclude_ids=hist)
    elif model_name == 'two_tower':
        user_emb_matrix = np.load('artifacts/two_tower/user_emb.npy')
        retriever = FAISSRetriever(
            'artifacts/two_tower/item_emb.npy',
            'artifacts/two_tower/faiss.index',
        )
        retrieve_fn = lambda uid, hist: retriever.retrieve(user_emb_matrix[uid], k=top_k, exclude_ids=hist)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    val_dict = val_df.set_index('user_id')['item_id'].to_dict()
    user_ids = val_df['user_id'].unique()

    chunks = []
    for i, uid in enumerate(user_ids):
        history = user_history.get(int(uid), [])
        gt = int(val_dict[uid])
        if not history and model_name in ('cf', 'svd'):
            continue
        cids, cscores = retrieve_fn(int(uid), history)
        if len(cids) == 0:
            continue
        feat = build_features(int(uid), cids, cscores,
                              user_demo, n_history, user_genre, item_genre, item_pop)
        feat['label'] = (feat['item_id'] == gt).astype(np.int8)
        feat['model'] = model_name
        chunks.append(feat)
        if (i + 1) % 1000 == 0:
            logger.info(f'[{model_name}] {i+1}/{len(user_ids)} users')

    out_df = pd.concat(chunks, ignore_index=True)
    out_path = out_dir / f'reranker_train_{model_name}.parquet'
    out_df.to_parquet(out_path, index=False)
    pos = int(out_df['label'].sum())
    logger.info(f'[{model_name}] Dataset: {len(out_df)} rows, {pos} positives -> {out_path}')
    return out_df


if __name__ == '__main__':
    for m in ['cf', 'svd', 'two_tower']:
        build_reranker_dataset(m)
