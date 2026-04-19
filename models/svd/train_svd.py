import scipy.sparse as sp
import pandas as pd
import numpy as np
import time
import wandb
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.svd.als_model import ALSModel
from evaluation.metrics import evaluate_model, evaluate_vectorized


logger = get_logger('train_svd')


CF_BASELINE_HR10 = 0.0830  # UserUserCF K=50 from Section 3


def train_run(item_user, val_df, train_df, factors, regularization, cfg):
    run_name = f'als_f{factors}_r{regularization}'
    wandb.init(
        project=cfg['wandb']['project'],
        group='svd',
        name=run_name,
        config={'model': 'ALS', 'factors': factors,
                'regularization': regularization, 'iterations': 50},
        reinit=True,
    )
    t0 = time.time()
    model = ALSModel(factors=factors, regularization=regularization, iterations=50)
    model.fit(item_user)
    fit_time = time.time() - t0

    score_matrix = model.score_all_users()
    metrics = evaluate_vectorized(score_matrix, val_df, train_df, K_list=[5, 10, 20])
    metrics['fit_time_s'] = fit_time
    wandb.log(metrics)
    logger.info(f'{run_name}: {metrics}')
    wandb.finish()
    return model, metrics


def benchmark_fold_in(best_model, train_df, val_df):
    """Measure fold-in latency on 100 users with simulated 1-5 interaction histories."""
    sparse_users = train_df.groupby('user_id').filter(lambda x: len(x) <= 5)
    if len(sparse_users) == 0:
        # ML-1M all users have 20+ — sample 100 warm users and thin to 3 items each
        sample_users = np.random.choice(train_df['user_id'].unique(), 100, replace=False)
        sparse_users = train_df[train_df['user_id'].isin(sample_users)]
        sparse_users = sparse_users.groupby('user_id').head(3)

    latencies = []
    for uid, group in list(sparse_users.groupby('user_id'))[:100]:
        item_ids = group['item_id'].tolist()
        t0 = time.time()
        emb = best_model.fold_in(item_ids)
        best_model.recommend_from_embedding(emb, item_ids, N=10)
        latencies.append((time.time() - t0) * 1000)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    logger.info(f'Fold-in latency: p50={p50:.1f}ms, p95={p95:.1f}ms')
    return p50, p95


def run():
    set_seed()
    cfg = get_config()
    cache = Path(cfg['data']['cache_dir'])
    processed = Path(cfg['data']['processed_dir'])
    models_dir = Path('artifacts/models/svd')
    models_dir.mkdir(parents=True, exist_ok=True)

    item_user = sp.load_npz(cache / 'svd_item_user_matrix.npz')
    val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
    train_df = pd.read_parquet(processed / 'train.parquet')
    logger.info(f'item_user: {item_user.shape}')

    svd_cfg = cfg['models']['svd']
    best_hr10 = 0.0
    best_model = None
    best_params = None

    for factors in svd_cfg['factors']:
        for reg in svd_cfg['regularization']:
            model, metrics = train_run(item_user, val_df, train_df, factors, reg, cfg)
            if metrics['HR@10'] > best_hr10:
                best_hr10 = metrics['HR@10']
                best_model = model
                best_params = {'factors': factors, 'regularization': reg}

    best_model.save(str(models_dir / 'als_best.pkl'))
    logger.info(f'Best ALS: {best_params}, HR@10={best_hr10:.4f}')

    p50, p95 = benchmark_fold_in(best_model, train_df, val_df)

    wandb.init(
        project=cfg['wandb']['project'],
        group='svd', name='svd_summary',
        config=best_params, reinit=True,
    )
    wandb.log({
        'best_HR@10': best_hr10,
        'fold_in_p50_ms': p50,
        'fold_in_p95_ms': p95,
        'cf_baseline_HR@10': CF_BASELINE_HR10,
    })
    wandb.finish()
    logger.info('Section 4 SVD complete.')


if __name__ == '__main__':
    run()
