import json
import numpy as np
import pandas as pd
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('tt_feature_prep')


def build_item_feature_matrices(cfg):
    """
    Builds two aligned matrices of shape (n_items, *) where n_items=3952.
    content_emb: (3952, 384) — ST embeddings, zeros for missing items
    genre_matrix: (3952, 18) — multi-hot genre vectors, zeros for missing
    """
    processed = Path(cfg['data']['processed_dir'])
    emb_dir = Path(cfg['models']['two_tower']['embeddings_path'])
    out_dir = Path('artifacts/two_tower')
    out_dir.mkdir(parents=True, exist_ok=True)

    st_emb = np.load(emb_dir / 'item_title_embeddings.npy')  # (3883, 384)
    item_meta = pd.read_parquet(processed / 'item_metadata.parquet')
    assert len(item_meta) == len(st_emb), 'ST embeddings and item_metadata row count mismatch'

    stats = json.load(open(processed / 'dataset_stats.json'))
    n_items = stats['n_items']
    n_genres = stats['n_genres']

    content_emb = np.zeros((n_items, st_emb.shape[1]), dtype=np.float32)
    genre_matrix = np.zeros((n_items, n_genres), dtype=np.float32)

    for idx, row in item_meta.iterrows():
        iid = int(row['item_id'])
        if iid < n_items:
            content_emb[iid] = st_emb[idx]
            genre_matrix[iid] = row['genre_vector']

    n_filled = (content_emb.sum(axis=1) != 0).sum()
    logger.info(f'Content emb: {content_emb.shape}, filled={n_filled}/{n_items}')
    logger.info(f'Genre matrix: {genre_matrix.shape}')

    np.save(out_dir / 'item_content_emb.npy', content_emb)
    np.save(out_dir / 'item_genre_matrix.npy', genre_matrix)
    return content_emb, genre_matrix


if __name__ == '__main__':
    cfg = get_config()
    build_item_feature_matrices(cfg)
