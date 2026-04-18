import pandas as pd
import numpy as np
import json
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('preprocess')


def load_raw(cfg):
    path = Path(cfg['data']['raw_dir']) / 'electronics.parquet'
    return pd.read_parquet(path)


def clean(df, cfg):
    logger.info('Cleaning...')
    df = df.drop_duplicates(subset=['user_id', 'item_id_raw', 'timestamp'])
    df = df.dropna(subset=['user_id', 'item_id_raw', 'timestamp', 'rating'])
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['timestamp'] = df['timestamp'].astype('int64')
    df['rating'] = df['rating'].astype('float32')
    df['title'] = df['title'].fillna('Unknown')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df


def filter_interactions(df, cfg):
    logger.info('Filtering by min interactions...')
    item_counts = df.groupby('item_id_raw').size()
    valid_items = item_counts[item_counts >= cfg['data']['min_item_interactions']].index
    df = df[df['item_id_raw'].isin(valid_items)]


    user_counts = df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= cfg['data']['min_user_interactions']].index
    df = df[df['user_id'].isin(valid_users)]
    logger.info(f'After filter: {df.shape}, users: {df.user_id.nunique()}, items: {df.item_id_raw.nunique()}')
    return df


def encode_ids(df):
    logger.info('Encoding user/item IDs...')
    user_map = {u: i for i, u in enumerate(sorted(df['user_id'].unique()))}
    item_map = {it: i for i, it in enumerate(sorted(df['item_id_raw'].unique()))}
    df['user_id'] = df['user_id'].map(user_map).astype('int64')
    df['item_id'] = df['item_id_raw'].map(item_map).astype('int64')
    return df, user_map, item_map


def add_features(df, cfg):
    df['implicit_feedback'] = 1
    df['user_history_len'] = df.groupby('user_id')['user_id'].transform('count')
    # Price bucket: 5 quantiles (0-4), NaN -> -1
    df['price_bucket'] = pd.qcut(df['price'], q=5, labels=False, duplicates='drop')
    df['price_bucket'] = df['price_bucket'].fillna(-1).astype('int8')
    # Popularity rank: normalized item interaction count
    item_pop = df.groupby('item_id').size().rename('item_pop')
    df = df.merge(item_pop, on='item_id')
    df['popularity_rank'] = (df['item_pop'] / df['item_pop'].max()).astype('float32')
    df = df.drop(columns=['item_pop'])
    return df


def temporal_split(df, cfg):
    logger.info('Temporal split — strictly by timestamp, no leakage...')
    df = df.sort_values('timestamp')
    n = len(df)
    train_end = int(n * cfg['data']['train_ratio'])
    val_end = int(n * (cfg['data']['train_ratio'] + cfg['data']['val_ratio']))
    df['split'] = 'test'
    df.iloc[:train_end, df.columns.get_loc('split')] = 'train'
    df.iloc[train_end:val_end, df.columns.get_loc('split')] = 'val'
    logger.info(f"Split sizes — train: {(df.split=='train').sum()}, val: {(df.split=='val').sum()}, test: {(df.split=='test').sum()}")
    return df


def save_outputs(df, user_map, item_map, cfg):
    out = Path(cfg['data']['processed_dir'])
    out.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        subset = df[df['split'] == split].drop(columns=['split'])
        subset.to_parquet(out / f'{split}.parquet', index=False)
        logger.info(f'Saved {split}: {subset.shape}')
    df.to_parquet(out / 'full.parquet', index=False)
    # Save mappings for inference
    with open(out / 'user_map.json', 'w') as f: json.dump({str(k): v for k, v in user_map.items()}, f)
    with open(out / 'item_map.json', 'w') as f: json.dump({str(k): v for k, v in item_map.items()}, f)
    # Save item metadata for two-tower
    item_meta = df[['item_id', 'item_id_raw', 'title', 'price_bucket', 'popularity_rank', 'main_category']].drop_duplicates(subset='item_id')
    item_meta.to_parquet(out / 'item_metadata.parquet', index=False)
    logger.info(f'Saved item metadata: {item_meta.shape}')


def run():
    cfg = get_config()
    df = load_raw(cfg)
    df = clean(df, cfg)
    df = filter_interactions(df, cfg)
    df, user_map, item_map = encode_ids(df)
    df = add_features(df, cfg)
    df = temporal_split(df, cfg)
    save_outputs(df, user_map, item_map, cfg)
    logger.info('Preprocessing complete.')
    return df


if __name__ == '__main__':
    run()
