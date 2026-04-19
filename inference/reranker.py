"""LightGBM binary reranker. Trains on combined CF+SVD+Two-Tower data."""
import pickle
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utils.logger import get_logger
from utils.seed import set_seed


logger = get_logger('reranker')


GENRES = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']

FEATURE_COLS = [
    'retrieval_score', 'gender_enc', 'age_enc', 'occupation',
    'n_history', 'genre_affinity', 'item_popularity',
] + [f'user_genre_{g}' for g in GENRES]


def train_reranker():
    set_seed()
    reranker_dir = Path('artifacts/reranker')
    reranker_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for m in ['cf', 'svd', 'two_tower']:
        p = reranker_dir / f'reranker_train_{m}.parquet'
        if p.exists():
            d = pd.read_parquet(p)
            dfs.append(d)
            logger.info(f'Loaded {m}: {len(d)} rows, {int(d["label"].sum())} positives')
    df = pd.concat(dfs, ignore_index=True)
    pos = int(df['label'].sum())
    logger.info(f'Combined: {len(df)} rows, {pos} positives ({pos/len(df):.3%})')

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['label'].values.astype(np.int32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y,
    )

    n_neg = int((y_tr == 0).sum())
    n_pos = max(int((y_tr == 1).sum()), 1)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(50)],
    )

    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    logger.info(f'Reranker val AUC: {auc:.4f}')

    imp = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False)
    imp.to_csv(reranker_dir / 'feature_importance.csv', index=False)
    logger.info(f'Top features:\n{imp.head().to_string()}')

    with open(reranker_dir / 'reranker.pkl', 'wb') as f:
        pickle.dump(clf, f)
    logger.info('Saved reranker.pkl')
    return clf, auc


class Reranker:
    """Inference wrapper. Reranks K candidates -> top-N."""

    def __init__(self, path='artifacts/reranker/reranker.pkl'):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info('Reranker loaded')

    def rerank(self, candidates_df: pd.DataFrame, k: int = 10) -> list:
        X = candidates_df[FEATURE_COLS].values.astype(np.float32)
        probs = self.model.predict_proba(X)[:, 1]
        top_idx = np.argsort(probs)[::-1][:k]
        return candidates_df.iloc[top_idx]['item_id'].astype(int).tolist()


if __name__ == '__main__':
    train_reranker()
