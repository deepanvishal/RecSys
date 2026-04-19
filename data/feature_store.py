import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed
from utils.device import get_device, log_device_info


logger = get_logger('feature_store')


def build_cf_matrix(cfg):
    train = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'train.parquet')
    full = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'full.parquet')
    n_users = int(full['user_id'].max()) + 1
    n_items = int(full['item_id'].max()) + 1
    mat = sp.csr_matrix(
        (train['implicit_feedback'].values,
         (train['user_id'].values, train['item_id'].values)),
        shape=(n_users, n_items), dtype=np.float32,
    )
    cache = Path(cfg['data']['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    sp.save_npz(cache / 'cf_interaction_matrix.npz', mat)
    logger.info(f'CF matrix: {mat.shape}, nnz={mat.nnz}')
    return mat


def build_svd_matrix(cfg):
    train = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'train.parquet')
    full = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'full.parquet')
    n_users = int(full['user_id'].max()) + 1
    n_items = int(full['item_id'].max()) + 1
    mat = sp.csr_matrix(
        (train['implicit_feedback'].values,
         (train['user_id'].values, train['item_id'].values)),
        shape=(n_users, n_items), dtype=np.float32,
    )
    item_user = mat.T.tocsr()
    sp.save_npz(Path(cfg['data']['cache_dir']) / 'svd_item_user_matrix.npz', item_user)
    logger.info(f'SVD item-user matrix: {item_user.shape}')
    return item_user


def build_st_embeddings(cfg):
    set_seed()
    item_meta = pd.read_parquet(Path(cfg['data']['processed_dir']) / 'item_metadata.parquet')
    model_name = cfg['models']['two_tower']['sentence_transformer_model']
    device = get_device()
    log_device_info()
    logger.info(f'Encoding {len(item_meta)} movie titles on {device}...')
    st = SentenceTransformer(model_name, device=str(device))
    texts = (item_meta['title'].fillna('') + ' ' + item_meta['genres'].fillna('')).tolist()
    embeddings = st.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    emb_dir = Path(cfg['models']['two_tower']['embeddings_path'])
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / 'item_title_embeddings.npy', embeddings)
    logger.info(f'ST embeddings saved: {embeddings.shape}  (~{embeddings.nbytes/1e6:.1f} MB)')
    return embeddings


def run():
    cfg = get_config()
    build_cf_matrix(cfg)
    build_svd_matrix(cfg)
    build_st_embeddings(cfg)
    logger.info('Feature store complete.')


if __name__ == '__main__':
    run()
