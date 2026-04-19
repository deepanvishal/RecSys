import pandas as pd
from config.config_loader import get_config
from inference.engine import TieredRecommendationEngine
from evaluation.metrics import evaluate_model


def _build_user_history(train_df):
    """Precompute user_id -> chronological item list. Avoids per-user DataFrame scans."""
    return (
        train_df.sort_values('timestamp')
        .groupby('user_id', sort=False)['item_id']
        .apply(list).to_dict()
    )


def evaluate_tiers(engine, train_df, val_df, cohorts_df):
    user_hist = _build_user_history(train_df)
    results = {}

    # Tier 3 — warm users (full history)
    warm_ids = set(cohorts_df[cohorts_df['cohort'] == 'warm']['user_id'].values.tolist())
    warm_val = val_df[val_df['user_id'].isin(warm_ids)]

    def sasrec_recommend(uid, N):
        hist = user_hist.get(int(uid), [])
        return engine.recommend(int(uid), hist, N=N)['items']

    metrics_t3 = evaluate_model(sasrec_recommend, warm_val)
    results['tier3_sasrec'] = metrics_t3
    print(f'Tier 3 (SASRec, warm users, n={len(warm_val)}): HR@10={metrics_t3["HR@10"]:.4f} NDCG@10={metrics_t3["NDCG@10"]:.4f}')

    # Tier 2 — sparse users (synthetically thinned to first 3 items)
    sparse_ids = set(
        cohorts_df[cohorts_df['cohort'].isin(['sparse_low', 'sparse_high'])]
        ['user_id'].values.tolist()
    )
    sparse_val = val_df[val_df['user_id'].isin(sparse_ids)]

    def svd_recommend(uid, N):
        hist = user_hist.get(int(uid), [])[:3]
        return engine.recommend(int(uid), hist, N=N)['items']

    metrics_t2 = evaluate_model(svd_recommend, sparse_val)
    results['tier2_svd'] = metrics_t2
    print(f'Tier 2 (SVD fold-in, sparse, n={len(sparse_val)}): HR@10={metrics_t2["HR@10"]:.4f} NDCG@10={metrics_t2["NDCG@10"]:.4f}')

    return results


if __name__ == '__main__':
    cfg = get_config()
    engine = TieredRecommendationEngine(cfg)
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/val.parquet')[['user_id', 'item_id']]
    cohorts = pd.read_parquet('data/processed/cohorts/user_cohorts.parquet')
    evaluate_tiers(engine, train_df, val_df, cohorts)
