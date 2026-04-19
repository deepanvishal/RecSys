import pandas as pd
import numpy as np
import json
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed


logger = get_logger('simulate')


def _to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, list): return [_to_native(i) for i in obj]
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    return obj


def create_sparse_cohorts(cfg):
    set_seed()
    sim = cfg['simulation']
    thresholds = sim['sparse_cohorts']
    train = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'train.parquet')
    hist = train.groupby('user_id').size().rename('history_len').reset_index()

    warm = hist[hist['history_len'] > thresholds['sparse_high']].copy()
    warm['cohort'] = 'warm'
    cohorts = [warm]
    warm_ids = warm['user_id'].values

    for name, cap in [('sparse_high', thresholds['sparse_high']),
                       ('sparse_low', thresholds['sparse_low']),
                       ('cold', 0)]:
        n = min(sim['min_users_per_cohort'], len(warm_ids))
        ids = np.random.choice(warm_ids, size=n, replace=False)
        row = pd.DataFrame({'user_id': ids, 'history_len': cap, 'cohort': name})
        cohorts.append(row)
        logger.info(f'Cohort {name}: {n} users')

    out_df = pd.concat(cohorts, ignore_index=True)
    out_path = Path(cfg['data']['processed_dir']) / 'cohorts'
    out_path.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path / 'user_cohorts.parquet', index=False)
    return out_df


def simulate_daily_updates(df_full, cfg):
    set_seed()
    sim = cfg['simulation']
    items = df_full['item_id'].unique()
    max_item = int(df_full['item_id'].max())
    max_user = int(df_full['user_id'].max())
    active = set(items.tolist())
    snaps = []

    for day in range(1, sim['simulation_days'] + 1):
        n_dep = max(1, int(len(active) * sim['daily_deprecation_rate']))
        dep = set(np.random.choice(list(active), size=n_dep, replace=False).tolist())
        active -= dep
        n_new = max(1, int(len(items) * sim['daily_new_item_rate']))
        new_items = list(range(max_item + 1, max_item + 1 + n_new))
        max_item += n_new
        active.update(new_items)
        new_users = list(range(max_user + 1, max_user + 1 + sim['daily_new_users']))
        max_user += sim['daily_new_users']
        snaps.append(_to_native({
            'day': day, 'n_active': len(active),
            'n_deprecated': n_dep, 'n_new_items': n_new,
            'n_new_users': sim['daily_new_users'],
            'new_user_ids': new_users, 'new_item_ids': new_items,
            'deprecated_item_ids': list(dep),
        }))

    out = Path(cfg['data']['processed_dir']) / 'simulation'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'daily_snapshots.json', 'w') as f:
        json.dump(snaps, f, indent=2)
    logger.info(f'Simulated {sim["simulation_days"]} days.')
    return snaps


def run():
    cfg = get_config()
    full = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'full.parquet')
    create_sparse_cohorts(cfg)
    simulate_daily_updates(full, cfg)


if __name__ == '__main__':
    run()
