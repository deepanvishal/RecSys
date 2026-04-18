import pandas as pd
import numpy as np
import json
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed


logger = get_logger('simulate')


def create_sparse_cohorts(df, cfg):
    """
    Thin existing warm users to create balanced cohorts.
    Returns a DataFrame with cohort column added.
    """
    set_seed()
    sim_cfg = cfg['simulation']
    thresholds = sim_cfg['sparse_cohorts']
    min_per_cohort = sim_cfg['min_users_per_cohort']


    train = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'train.parquet')
    user_hist = train.groupby('user_id').size().rename('history_len').reset_index()


    def assign_cohort(h):
        if h == thresholds['cold']: return 'cold'
        elif h <= thresholds['sparse_low']: return 'sparse_low'
        elif h <= thresholds['sparse_high']: return 'sparse_high'
        else: return 'warm'


    user_hist['cohort'] = user_hist['history_len'].apply(assign_cohort)


    cohort_dfs = []
    for cohort_name in ['cold', 'sparse_low', 'sparse_high', 'warm']:
        cohort_users = user_hist[user_hist['cohort'] == cohort_name]
        if cohort_name == 'cold':
            # Synthesize cold users (no history)
            n = max(min_per_cohort, len(cohort_users))
            warm_users = user_hist[user_hist['cohort'] == 'warm']['user_id'].values
            sampled = np.random.choice(warm_users, size=n, replace=False)
            cold_df = pd.DataFrame({'user_id': sampled, 'history_len': 0, 'cohort': 'cold'})
            cohort_dfs.append(cold_df)
        else:
            cohort_dfs.append(cohort_users)
        logger.info(f'Cohort {cohort_name}: {len(cohort_dfs[-1])} users')


    cohorts = pd.concat(cohort_dfs, ignore_index=True)


    out = Path(cfg['data']['processed_dir']) / 'cohorts'
    out.mkdir(parents=True, exist_ok=True)
    cohorts.to_parquet(out / 'user_cohorts.parquet', index=False)
    logger.info(f'Saved cohorts to {out}/user_cohorts.parquet')
    return cohorts


def simulate_daily_updates(df_full, cfg):
    """
    Simulate daily catalog and user changes over N days.
    Saves one snapshot parquet per day.
    """
    set_seed()
    sim_cfg = cfg['simulation']
    n_days = sim_cfg['simulation_days']
    new_users_per_day = sim_cfg['daily_new_users']
    deprecation_rate = sim_cfg['daily_deprecation_rate']
    new_item_rate = sim_cfg['daily_new_item_rate']


    out = Path(cfg['data']['processed_dir']) / 'simulation'
    out.mkdir(parents=True, exist_ok=True)


    all_items = df_full['item_id'].unique()
    max_user_id = df_full['user_id'].max()
    max_item_id = df_full['item_id'].max()
    active_items = set(all_items)
    deprecated_items = set()


    snapshots = []
    for day in range(1, n_days + 1):
        # Deprecate items
        n_deprecate = max(1, int(len(active_items) * deprecation_rate))
        to_deprecate = set(np.random.choice(list(active_items), size=n_deprecate, replace=False))
        active_items -= to_deprecate
        deprecated_items |= to_deprecate


        # Add new items
        n_new_items = max(1, int(len(all_items) * new_item_rate))
        new_item_ids = list(range(max_item_id + 1, max_item_id + 1 + n_new_items))
        max_item_id += n_new_items
        active_items.update(new_item_ids)


        # Add new users
        new_user_ids = list(range(max_user_id + 1, max_user_id + 1 + new_users_per_day))
        max_user_id += new_users_per_day


        snapshot = {
            'day': day,
            'n_active_items': len(active_items),
            'n_deprecated_items': len(deprecated_items),
            'n_new_items_today': n_new_items,
            'n_new_users_today': new_users_per_day,
            'new_user_ids': new_user_ids,
            'new_item_ids': new_item_ids,
            'deprecated_item_ids': list(to_deprecate),
        }
        snapshots.append(snapshot)
        logger.info(f'Day {day}: active_items={len(active_items)}, new_users={new_users_per_day}, deprecated={n_deprecate}')


    # Save snapshots
    with open(out / 'daily_snapshots.json', 'w') as f:
        json.dump(snapshots, f, indent=2)
    logger.info(f'Saved {n_days} daily snapshots to {out}/daily_snapshots.json')
    return snapshots


def run():
    cfg = get_config()
    df_full = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'full.parquet')
    cohorts = create_sparse_cohorts(df_full, cfg)
    snapshots = simulate_daily_updates(df_full, cfg)
    logger.info('Simulation complete.')
    return cohorts, snapshots


if __name__ == '__main__':
    run()
