import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
from pathlib import Path
import wandb
import os
from config.config_loader import get_config, get_wandb_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.svd import SVDModel
from evaluation.metrics import evaluate_all


logger = get_logger('train_svd')


# CF baseline from Section 3 — log this in every run for comparison
CF_BASELINE_HIT10 = 0.009  # 0.9%


def load_data(cfg):
    processed = Path(cfg['data']['processed_dir'])
    cache = Path(cfg['data']['cache_dir'])
    item_user_matrix = sp.load_npz(cache / 'svd_item_user_matrix.npz')
    train = pd.read_parquet(processed / 'train.parquet')
    val = pd.read_parquet(processed / 'val.parquet')
    test = pd.read_parquet(processed / 'test.parquet')
    cohorts = pd.read_parquet(processed / 'cohorts' / 'user_cohorts.parquet')
    return item_user_matrix, train, val, test, cohorts


def evaluate(model, interactions_df, cohorts, cfg, use_foldin=False):
    """
    Evaluate model across all users and cohorts.
    use_foldin=True uses fold-in path for sparse cohorts (realistic inference simulation).
    """
    k_values = cfg['evaluation']['k_values']
    primary_k = cfg['evaluation']['primary_k']
    n_test_users = cfg['evaluation']['n_test_users']

    test_users = interactions_df['user_id'].unique()
    if len(test_users) > n_test_users:
        np.random.shuffle(test_users)
        test_users = test_users[:n_test_users]

    ground_truth = (
        interactions_df.sort_values('timestamp')
        .groupby('user_id')['item_id'].last().to_dict()
    )

    # Precompute user -> item list once (avoid 1000 DataFrame scans)
    user_items_map = (
        interactions_df.groupby('user_id')['item_id'].apply(list).to_dict()
        if use_foldin else {}
    )

    sparse_users = set(cohorts[cohorts['cohort'].isin(['sparse_low', 'sparse_high'])]['user_id'])

    recommendations = {}
    for user_id in test_users:
        if user_id not in ground_truth:
            continue
        if use_foldin and user_id in sparse_users:
            user_items = user_items_map.get(user_id, [])
            recs = model.foldin_recommend(user_items, n=max(k_values))
        else:
            recs = model.recommend(int(user_id), n=max(k_values))
        recommendations[user_id] = [r[0] for r in recs]

    n_items = model.n_items
    metrics = evaluate_all(
        ground_truth={u: gt for u, gt in ground_truth.items() if u in recommendations},
        recommendations=recommendations,
        k_values=k_values,
        n_total_items=n_items,
        popularity_scores=model.popularity_scores,
    )

    for cohort_name in ['cold', 'sparse_low', 'sparse_high', 'warm']:
        cohort_users = set(cohorts[cohorts['cohort'] == cohort_name]['user_id'])
        cohort_recs = {u: r for u, r in recommendations.items() if u in cohort_users}
        cohort_gt = {u: ground_truth[u] for u in cohort_recs if u in ground_truth}
        if not cohort_gt:
            metrics[f'hit@{primary_k}_{cohort_name}'] = 0.0
            continue
        hits = [1.0 if cohort_gt[u] in cohort_recs[u][:primary_k] else 0.0 for u in cohort_gt]
        metrics[f'hit@{primary_k}_{cohort_name}'] = float(np.mean(hits))

    return metrics


def demo_incremental_update(model, train, cfg):
    """
    Simulate one day of incremental updates:
    - Add N new users via fold-in (no retrain)
    - Log per-user fold-in latency
    """
    sim_cfg = cfg['simulation']
    n_new_users = sim_cfg['daily_new_users']
    logger.info(f'Simulating incremental update: {n_new_users} new users via fold-in')

    warm_users = train[train['user_history_len'] > 5]['user_id'].unique()
    sample_users = np.random.choice(warm_users, size=min(n_new_users, len(warm_users)), replace=False)

    # Precompute user -> items once
    sample_set = set(sample_users.tolist())
    sample_items = (
        train[train['user_id'].isin(sample_set)]
        .groupby('user_id')['item_id'].apply(list).to_dict()
    )

    foldin_times = []
    for uid in sample_users:
        items = sample_items.get(uid, [])
        t0 = time.time()
        _ = model.foldin_recommend(items, n=10)
        foldin_times.append(time.time() - t0)

    avg_foldin_ms = float(np.mean(foldin_times)) * 1000
    total_foldin_ms = float(np.sum(foldin_times)) * 1000

    logger.info(f'Fold-in avg latency: {avg_foldin_ms:.2f}ms per user')
    logger.info(f'Total fold-in for {n_new_users} users: {total_foldin_ms:.0f}ms')

    return {
        'incremental/avg_foldin_latency_ms': avg_foldin_ms,
        'incremental/total_foldin_ms': total_foldin_ms,
        'incremental/n_new_users': n_new_users,
    }


def train_and_evaluate():
    set_seed()
    cfg = get_config()
    wandb_cfg = get_wandb_config()
    item_user_matrix, train, val, test, cohorts = load_data(cfg)

    factors_list = cfg['models']['svd']['factors']
    save_path = cfg['models']['svd']['save_path']

    best_hit = -1.0
    best_model = None
    best_factors = None
    primary_k = cfg['evaluation']['primary_k']

    for factors in factors_list:
        run_name = f'svd_factors{factors}'
        logger.info(f'Training: {run_name}')

        wandb.init(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name=run_name,
            tags=cfg['wandb']['tags'] + ['svd'],
            config={
                'model': 'svd',
                'factors': factors,
                'regularization': cfg['models']['svd']['regularization'],
                'iterations': cfg['models']['svd']['iterations'],
                'dataset': cfg['data']['category'],
                'seed': cfg['project']['seed'],
                'cf_baseline_hit10': CF_BASELINE_HIT10,
            },
            reinit=True,
        )

        model = SVDModel(factors=factors)

        t0 = time.time()
        model.fit(item_user_matrix)
        train_time = time.time() - t0
        logger.info(f'Training time: {train_time:.1f}s')

        if hasattr(model.model, 'training_loss'):
            for i, loss in enumerate(model.model.training_loss):
                wandb.log({'train/loss': loss, 'iteration': i})

        wandb.log({
            'train/time_seconds': train_time,
            'n_users': model.n_users,
            'n_items': model.n_items,
            'cf_baseline_hit10': CF_BASELINE_HIT10,
        })

        val_metrics = evaluate(model, val, cohorts, cfg, use_foldin=False)
        wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
        logger.info(f'Val metrics (standard): {val_metrics}')

        val_foldin_metrics = evaluate(model, val, cohorts, cfg, use_foldin=True)
        wandb.log({f'val_foldin/{k}': v for k, v in val_foldin_metrics.items()})
        logger.info(f'Val metrics (fold-in): {val_foldin_metrics}')

        incremental_metrics = demo_incremental_update(model, train, cfg)
        wandb.log(incremental_metrics)

        if val_metrics[f'hit@{primary_k}'] > best_hit:
            best_hit = val_metrics[f'hit@{primary_k}']
            best_model = model
            best_factors = factors

        model.save(save_path)
        wandb.finish()

    logger.info(f'Best SVD: factors={best_factors}, val hit@{primary_k}={best_hit:.4f}')
    if CF_BASELINE_HIT10 > 0:
        logger.info(f'Improvement over CF baseline: {(best_hit - CF_BASELINE_HIT10) / CF_BASELINE_HIT10 * 100:.1f}%')

    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name=f'svd_factors{best_factors}_test_eval',
        tags=cfg['wandb']['tags'] + ['svd', 'test', 'best'],
        reinit=True,
    )
    test_metrics = evaluate(best_model, test, cohorts, cfg, use_foldin=True)
    wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
    wandb.log({
        'test/cf_baseline_hit10': CF_BASELINE_HIT10,
        'test/improvement_over_cf_pct': (test_metrics[f'hit@{primary_k}'] - CF_BASELINE_HIT10) / CF_BASELINE_HIT10 * 100,
    })
    logger.info(f'Test metrics: {test_metrics}')
    wandb.finish()

    best_model.save(save_path + 'best/')
    logger.info('SVD training complete.')
    return best_model, test_metrics


if __name__ == '__main__':
    train_and_evaluate()
