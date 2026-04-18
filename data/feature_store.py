import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed
from utils.device import get_device, log_device_info


logger = get_logger('feature_store')


def build_interaction_matrix(df, n_users, n_items):
    """COO sparse matrix: rows=users, cols=items, values=implicit_feedback"""
    return sp.csr_matrix(
        (df['implicit_feedback'].values, (df['user_id'].values, df['item_id'].values)),
        shape=(n_users, n_items),
        dtype=np.float32
    )


def build_cf_features(cfg):
    logger.info('Building CF features (sparse interaction matrix)...')
    train = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'train.parquet')
    full = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'full.parquet')
    n_users = full['user_id'].max() + 1
    n_items = full['item_id'].max() + 1
    matrix = build_interaction_matrix(train, n_users, n_items)
    cache_dir = Path(cfg['data']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(cache_dir / 'cf_interaction_matrix.npz', matrix)
    logger.info(f'CF matrix shape: {matrix.shape}, nnz: {matrix.nnz}')
    return matrix


def build_svd_features(cfg):
    logger.info('Building SVD features (user-item matrix for implicit ALS)...')
    train = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'train.parquet')
    full = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'full.parquet')
    n_users = full['user_id'].max() + 1
    n_items = full['item_id'].max() + 1
    # implicit lib expects item x user matrix
    matrix = build_interaction_matrix(train, n_users, n_items)
    item_user_matrix = matrix.T.tocsr()
    cache_dir = Path(cfg['data']['cache_dir'])
    sp.save_npz(cache_dir / 'svd_item_user_matrix.npz', item_user_matrix)
    logger.info(f'SVD item-user matrix shape: {item_user_matrix.shape}')
    return item_user_matrix


def build_two_tower_features(cfg):
    """Precompute sentence transformer embeddings for item titles."""
    set_seed()
    logger.info('Building Two-Tower features (title embeddings)...')
    item_meta = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'item_metadata.parquet')
    model_name = cfg['models']['two_tower']['sentence_transformer_model']
    device = get_device()
    log_device_info()
    logger.info(f'Loading sentence transformer: {model_name} on {device}')
    st_model = SentenceTransformer(model_name, device=str(device))
    titles = item_meta['title'].fillna('Unknown').tolist()
    logger.info(f'Encoding {len(titles)} item titles...')
    batch_size = 2048 if device.type == 'cuda' else 512
    embeddings = st_model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    emb_dir = Path(cfg['models']['two_tower']['embeddings_path'])
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / 'item_title_embeddings.npy', embeddings)
    item_meta.to_parquet(Path(cfg['data']['processed_dir']) / 'features_two_tower.parquet', index=False)
    logger.info(f'Saved title embeddings: {embeddings.shape}')
    return embeddings, item_meta


def run():
    cfg = get_config()
    build_cf_features(cfg)
    build_svd_features(cfg)
    build_two_tower_features(cfg)
    logger.info('Feature store build complete.')


if __name__ == '__main__':
    run()
