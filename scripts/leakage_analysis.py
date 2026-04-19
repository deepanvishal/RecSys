"""
scripts/leakage_analysis.py
Quantifies how many test users and items are unseen in training.
Documents the root cause of val hit@10=1.1% → test hit@10=0% collapse.
"""
import pandas as pd
import json
import wandb
from pathlib import Path
from config.config_loader import get_config, get_wandb_config
from utils.logger import get_logger


logger = get_logger('leakage_analysis')


def run():
    cfg = get_config()
    wandb_cfg = get_wandb_config()
    processed = Path(cfg['data']['processed_dir'])

    train = pd.read_parquet(processed / 'train.parquet')
    val = pd.read_parquet(processed / 'val.parquet')
    test = pd.read_parquet(processed / 'test.parquet')

    train_users = set(train['user_id'].unique())
    train_items = set(train['item_id'].unique())
    val_users = set(val['user_id'].unique())
    val_items = set(val['item_id'].unique())
    test_users = set(test['user_id'].unique())
    test_items = set(test['item_id'].unique())

    results = {
        'n_train_users': len(train_users),
        'n_val_users': len(val_users),
        'n_test_users': len(test_users),
        'val_users_seen_in_train': len(val_users & train_users),
        'val_users_seen_pct': len(val_users & train_users) / len(val_users) * 100,
        'test_users_seen_in_train': len(test_users & train_users),
        'test_users_seen_pct': len(test_users & train_users) / len(test_users) * 100,
        'test_users_unseen_pct': (1 - len(test_users & train_users) / len(test_users)) * 100,
        'n_train_items': len(train_items),
        'n_val_items': len(val_items),
        'n_test_items': len(test_items),
        'val_items_seen_in_train': len(val_items & train_items),
        'val_items_seen_pct': len(val_items & train_items) / len(val_items) * 100,
        'test_items_seen_in_train': len(test_items & train_items),
        'test_items_seen_pct': len(test_items & train_items) / len(test_items) * 100,
        'test_items_unseen_pct': (1 - len(test_items & train_items) / len(test_items)) * 100,
    }

    print('\n=== Train/Test Leakage Analysis ===')
    print(f'Users seen in test that were in train: {results["test_users_seen_pct"]:.1f}%')
    print(f'Users in test NOT seen in train: {results["test_users_unseen_pct"]:.1f}%')
    print(f'Items seen in test that were in train: {results["test_items_seen_pct"]:.1f}%')
    print(f'Items in test NOT seen in train: {results["test_items_unseen_pct"]:.1f}%')
    print('This explains the val→test metric collapse for ID-based models (CF, SVD).')
    print('Content-based Two-Tower is robust to this — new items have ST embeddings.')

    report_dir = Path(cfg['paths']['report'])
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / 'leakage_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved leakage analysis to {report_dir}/leakage_analysis.json')

    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name='leakage_analysis',
        tags=cfg['wandb']['tags'] + ['analysis'],
        reinit=True,
    )
    wandb.log({f'leakage/{k}': v for k, v in results.items()})
    wandb.finish()
    return results


if __name__ == '__main__':
    run()
