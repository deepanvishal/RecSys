import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import wandb
import os
from config.config_loader import get_config, get_wandb_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.cf import CollaborativeFilter
from evaluation.metrics import hit_at_k, ndcg_at_k, catalog_coverage, popularity_bias_index


logger = get_logger('train_cf')


def load_data(cfg):
    processed = Path(cfg['data']['processed_dir'])
    matrix = sp.load_npz(Path(cfg['data']['cache_dir']) / 'cf_interaction_matrix.npz')
    val = pd.read_parquet(processed / 'val.parquet')
    test = pd.read_parquet(processed / 'test.parquet')
    cohorts = pd.read_parquet(processed / 'cohorts' / 'user_cohorts.parquet')
    return matrix, val, test, cohorts


def evaluate(model, interactions_df, cohorts, cfg, split_name='val'):
    """Evaluate model. Returns dict of metrics including per-cohort breakdown."""
    k_values = cfg['evaluation']['k_values']
    n_test_users = cfg['evaluation']['n_test_users']
    primary_k = cfg['evaluation']['primary_k']

    test_users = interactions_df['user_id'].unique()
    if len(test_users) > n_test_users:
        test_users = np.random.choice(test_users, n_test_users, replace=False)

    ground_truth = interactions_df.sort_values('timestamp').groupby('user_id')['item_id'].last().to_dict()

    results = {u: model.recommend(int(u), n=max(k_values)) for u in test_users if u in ground_truth}

    metrics = {}
    for k in k_values:
        hits, ndcgs = [], []
        for user_id, recs in results.items():
            gt = ground_truth[user_id]
            rec_items = [r[0] for r in recs[:k]]
            hits.append(hit_at_k(gt, rec_items, k))
            ndcgs.append(ndcg_at_k(gt, rec_items, k))
        metrics[f'hit@{k}'] = float(np.mean(hits))
        metrics[f'ndcg@{k}'] = float(np.mean(ndcgs))

    all_recs = [item for recs in results.values() for item, _ in recs[:primary_k]]
    n_items = model.n_items
    metrics['coverage'] = catalog_coverage(all_recs, n_items)
    metrics['popularity_bias'] = popularity_bias_index(all_recs, model.popularity_scores)

    for cohort_name in ['cold', 'sparse_low', 'sparse_high', 'warm']:
        cohort_users = set(cohorts[cohorts['cohort'] == cohort_name]['user_id'].values)
        cohort_results = {u: r for u, r in results.items() if u in cohort_users}
        if not cohort_results:
            metrics[f'hit@{primary_k}_{cohort_name}'] = 0.0
            continue
        cohort_hits = []
        for user_id, recs in cohort_results.items():
            gt = ground_truth.get(user_id)
            if gt is None:
                continue
            rec_items = [r[0] for r in recs[:primary_k]]
            cohort_hits.append(hit_at_k(gt, rec_items, primary_k))
        metrics[f'hit@{primary_k}_{cohort_name}'] = float(np.mean(cohort_hits)) if cohort_hits else 0.0

    return metrics


def train_and_evaluate():
    set_seed()
    cfg = get_config()
    wandb_cfg = get_wandb_config()
    matrix, val, test, cohorts = load_data(cfg)

    cf_cfg = cfg['models']['cf']
    neighborhood_sizes = cf_cfg['neighborhood_sizes']
    model_types = ['item', 'user']
    save_path = cf_cfg['save_path']

    best_hit = -1.0
    best_model = None
    best_run_name = None
    primary_k = cfg['evaluation']['primary_k']

    for model_type in model_types:
        for k in neighborhood_sizes:
            run_name = f'cf_{model_type}_k{k}'
            logger.info(f'Training: {run_name}')

            wandb.init(
                project=wandb_cfg['project'],
                entity=wandb_cfg['entity'],
                name=run_name,
                tags=cfg['wandb']['tags'] + ['cf', model_type],
                config={
                    'model': 'cf',
                    'model_type': model_type,
                    'k': k,
                    'dataset': cfg['data']['category'],
                    'seed': cfg['project']['seed'],
                },
                reinit=True,
            )

            model = CollaborativeFilter(model_type=model_type, k=k)
            model.fit(matrix)

            wandb.log({
                'n_users': model.n_users,
                'n_items': model.n_items,
                'matrix_nnz': matrix.nnz,
                'sparsity': 1 - matrix.nnz / (model.n_users * model.n_items),
            })

            val_metrics = evaluate(model, val, cohorts, cfg, split_name='val')
            wandb.log({f'val/{m}': v for m, v in val_metrics.items()})
            logger.info(f'Val metrics: {val_metrics}')

            if val_metrics[f'hit@{primary_k}'] > best_hit:
                best_hit = val_metrics[f'hit@{primary_k}']
                best_model = model
                best_run_name = run_name

            model.save(save_path)
            wandb.finish()

    logger.info(f'Best model: {best_run_name} with val hit@{primary_k}={best_hit:.4f}')
    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name=f'{best_run_name}_test_eval',
        tags=cfg['wandb']['tags'] + ['cf', 'test', 'best'],
        reinit=True,
    )
    test_metrics = evaluate(best_model, test, cohorts, cfg, split_name='test')
    wandb.log({f'test/{m}': v for m, v in test_metrics.items()})
    logger.info(f'Test metrics: {test_metrics}')
    wandb.finish()

    best_model.save(save_path + 'best/')
    logger.info('CF training complete.')
    return best_model, test_metrics


if __name__ == '__main__':
    train_and_evaluate()
