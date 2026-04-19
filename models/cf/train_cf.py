import scipy.sparse as sp
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.cf.item_item_cf import ItemItemCF
from models.cf.user_user_cf import UserUserCF
from evaluation.metrics import evaluate_model, evaluate_vectorized


logger = get_logger('train_cf')


def train_item_item(matrix, val_df, train_df, K, cfg):
    wandb.init(
        project=cfg['wandb']['project'],
        group='cf',
        name=f'item_item_K{K}',
        config={'model': 'ItemItemCF', 'K': K},
        reinit=True,
    )
    model = ItemItemCF(K=K)
    model.fit(matrix)
    # CosineRecommender exposes user_factors (user-item csr) and item_factors (item-item top-K csr)
    user_f = model.model.user_factors
    item_f = model.model.item_factors
    score_matrix = np.asarray((user_f @ item_f.T).todense()).astype(np.float32)
    metrics = evaluate_vectorized(score_matrix, val_df, train_df, K_list=[5, 10, 20])
    wandb.log(metrics)
    logger.info(f'ItemItemCF K={K}: {metrics}')
    wandb.finish()
    return model, metrics


def train_user_user(matrix, val_df, K, cfg):
    wandb.init(
        project=cfg['wandb']['project'],
        group='cf',
        name=f'user_user_K{K}',
        config={'model': 'UserUserCF', 'K': K},
        reinit=True,
    )
    model = UserUserCF(K=K)
    model.fit(matrix)
    metrics = evaluate_model(model.recommend, val_df)
    wandb.log(metrics)
    logger.info(f'UserUserCF K={K}: {metrics}')
    wandb.finish()
    return model, metrics


def run():
    set_seed()
    cfg = get_config()
    cache = Path(cfg['data']['cache_dir'])
    processed = Path(cfg['data']['processed_dir'])
    models_dir = Path('artifacts/models/cf')
    models_dir.mkdir(parents=True, exist_ok=True)

    matrix = sp.load_npz(cache / 'cf_interaction_matrix.npz')
    val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
    train_df = pd.read_parquet(processed / 'train.parquet')[['user_id', 'item_id']]
    logger.info(f'Matrix: {matrix.shape}, val users: {len(val_df)}')

    best_hr10 = 0
    best_model = None
    best_K = None
    for K in cfg['models']['cf']['K_values']:
        model, metrics = train_item_item(matrix, val_df, train_df, K, cfg)
        if metrics['HR@10'] > best_hr10:
            best_hr10 = metrics['HR@10']
            best_model = model
            best_K = K

    best_model.save(str(models_dir / 'item_item_best.pkl'))
    logger.info(f'Best ItemItemCF: K={best_K}, HR@10={best_hr10:.4f}')

    uu_model, uu_metrics = train_user_user(matrix, val_df, 50, cfg)

    wandb.init(
        project=cfg['wandb']['project'],
        group='cf',
        name='cf_summary',
        config={'best_K': best_K},
        reinit=True,
    )
    wandb.log({
        'best_item_item_HR@10': best_hr10,
        'best_K': best_K,
        'user_user_HR@10': uu_metrics['HR@10'],
    })
    wandb.finish()
    logger.info('Section 3 CF complete.')


if __name__ == '__main__':
    run()
