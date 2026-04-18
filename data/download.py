import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('download')


def download_amazon_electronics():
    cfg = get_config()
    raw_dir = Path(cfg['data']['raw_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / 'electronics.parquet'


    if out_path.exists():
        logger.info(f'Raw data already exists at {out_path}, skipping download.')
        return out_path


    logger.info('Loading Amazon Electronics 2023 from HuggingFace...')
    # Load interactions only (no review text) — keeps RAM tractable.
    # rating_only configs contain: user_id, parent_asin, rating, timestamp
    reviews = load_dataset(
        cfg['data']['dataset_name'],
        '0core_rating_only_Electronics',
        split='full',
        trust_remote_code=True
    )
    reviews_df = reviews.to_pandas()


    # Load metadata
    meta = load_dataset(
        cfg['data']['dataset_name'],
        'raw_meta_Electronics',
        split='full',
        trust_remote_code=True
    )
    meta_df = meta.to_pandas()


    # Keep only needed columns from reviews — join on parent_asin (product-level)
    reviews_df = reviews_df[['user_id', 'parent_asin', 'rating', 'timestamp']].copy()
    reviews_df.rename(columns={'parent_asin': 'item_id_raw'}, inplace=True)


    # Keep only needed columns from metadata
    meta_df = meta_df[['parent_asin', 'title', 'price', 'main_category']].copy()
    meta_df.rename(columns={'parent_asin': 'item_id_raw'}, inplace=True)
    meta_df = meta_df.drop_duplicates(subset='item_id_raw')


    # Merge
    df = reviews_df.merge(meta_df, on='item_id_raw', how='left')


    logger.info(f'Raw dataset shape: {df.shape}')
    logger.info(f'Unique users: {df.user_id.nunique()}')
    logger.info(f'Unique items: {df.item_id_raw.nunique()}')


    df.to_parquet(out_path, index=False)
    logger.info(f'Saved raw data to {out_path}')
    return out_path


if __name__ == '__main__':
    download_amazon_electronics()
