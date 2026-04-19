import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config.config_loader import get_config
from utils.logger import get_logger
from utils.device import get_device
from utils.seed import set_seed


logger = get_logger('bias_audit')


GENRES = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']


AGE_LABELS = {0: '<18', 1: '18-24', 2: '25-34', 3: '35-44',
              4: '45-49', 5: '50-55', 6: '56+'}


OCC_LABELS = {
    0: 'other', 1: 'academic/educator', 2: 'artist', 3: 'clerical/admin',
    4: 'college/grad student', 5: 'customer service', 6: 'doctor/health care',
    7: 'executive/managerial', 8: 'farmer', 9: 'homemaker', 10: 'K-12 student',
    11: 'lawyer', 12: 'programmer', 13: 'retired', 14: 'sales/marketing',
    15: 'scientist', 16: 'self-employed', 17: 'technician/engineer',
    18: 'tradesman/craftsman', 19: 'unemployed', 20: 'writer',
}


def get_sasrec_score_matrix(processed, artifacts, device):
    from models.sasrec.model import SASRec
    from models.sasrec.dataset import build_user_sequences
    from models.sasrec.train_sasrec import get_score_matrix
    stats = json.load(open(processed / 'dataset_stats.json'))
    n_users, n_items = stats['n_users'], stats['n_items']
    cfg = get_config()
    sr_cfg = cfg['models']['sasrec']
    model = SASRec(
        n_items=n_items, hidden=sr_cfg['hidden'],
        max_len=sr_cfg['max_len'], num_blocks=sr_cfg['num_blocks'],
        num_heads=sr_cfg['num_heads'], dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(
        str(artifacts / 'models/sasrec/sasrec_best.pt'), map_location=device,
    ))
    model.eval()
    train_df = pd.read_parquet(processed / 'train.parquet')
    user_seqs = build_user_sequences(train_df, max_len=sr_cfg['max_len'])
    return get_score_matrix(model, user_seqs, n_users, n_items, device), n_users, n_items


def get_svd_score_matrix(artifacts):
    from models.svd.als_model import ALSModel
    model = ALSModel.load(str(artifacts / 'models/svd/als_best.pkl'))
    return model.score_all_users()


def mask_seen(score_matrix, train_df):
    scores = score_matrix.copy()
    seen = train_df[['user_id', 'item_id']].values.astype(np.int64)
    scores[seen[:, 0], seen[:, 1]] = -np.inf
    return scores


def get_topk_recs(scores, K=10):
    return np.argsort(scores, axis=1)[:, ::-1][:, :K]


# ---- Popularity bias ----
def popularity_bias(recs, item_pop, n_items, K=10):
    """
    recs: (n_users, K) item IDs in [0, n_items).
    item_pop: Series indexed by item_id, values = train interaction count.
    """
    pop_arr = item_pop.reindex(range(n_items)).fillna(0).values  # (n_items,)
    rec_pops = pop_arr[recs]
    mean_rec_pop = rec_pops.mean()
    mean_item_pop = pop_arr[pop_arr > 0].mean()
    pop_ratio = mean_rec_pop / mean_item_pop
    median_pop = np.median(pop_arr[pop_arr > 0])
    longtail_frac = (rec_pops < median_pop).mean()
    return {
        'mean_rec_popularity': float(mean_rec_pop),
        'mean_item_popularity': float(mean_item_pop),
        'popularity_ratio': float(pop_ratio),
        'longtail_fraction': float(longtail_frac),
    }


# ---- Demographic bias ----
def demographic_bias(recs, val_df, user_meta, group_col, label_map=None):
    merged = val_df.merge(user_meta[['user_id', group_col]], on='user_id')
    rows = []
    for grp_val, grp_df in merged.groupby(group_col):
        uids = grp_df['user_id'].values.astype(int)
        gts = grp_df['item_id'].values.astype(int)
        grp_recs = recs[uids]
        hits = np.array([int(gt in grp_recs[i]) for i, gt in enumerate(gts)])
        label = label_map[grp_val] if label_map else str(grp_val)
        rows.append({
            'group': label,
            'n_users': len(uids),
            'HR@10': float(hits.mean()),
        })
    return pd.DataFrame(rows).sort_values('group')


# ---- Genre concentration ----
def genre_concentration(recs, genre_matrix):
    """
    genre_matrix: (n_items, 18) item-id-aligned multi-hot genre matrix
        (we use the precomputed artifacts/two_tower/item_genre_matrix.npy
        which is aligned to item_id 0..n_items-1; missing rows are zero).
    """
    rec_flat = recs.flatten()
    rec_flat = np.clip(rec_flat, 0, genre_matrix.shape[0] - 1)
    rec_genres = genre_matrix[rec_flat]            # (n_users*K, 18)
    genre_fracs = rec_genres.mean(axis=0)          # (18,)
    return pd.DataFrame(
        {'genre': GENRES, 'fraction_in_recs': genre_fracs},
    ).sort_values('fraction_in_recs', ascending=False)


def run():
    set_seed()
    device = get_device()
    processed = Path('data/processed')
    artifacts = Path('artifacts')
    out_dir = Path('artifacts/bias')
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(processed / 'train.parquet')
    val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
    user_meta = pd.read_parquet(processed / 'user_metadata.parquet')
    item_pop = train_df.groupby('item_id').size()
    genre_matrix = np.load(str(artifacts / 'two_tower/item_genre_matrix.npy'))

    # ---- SASRec ----
    logger.info('Computing SASRec score matrix...')
    sasrec_scores, n_users, n_items = get_sasrec_score_matrix(processed, artifacts, device)
    sasrec_scores = mask_seen(sasrec_scores, train_df)
    sasrec_recs = get_topk_recs(sasrec_scores, K=10)

    pop_bias = popularity_bias(sasrec_recs, item_pop, n_items)
    logger.info(f'SASRec popularity ratio: {pop_bias["popularity_ratio"]:.3f}')
    logger.info(f'SASRec long-tail fraction: {pop_bias["longtail_fraction"]:.3f}')
    pd.DataFrame([pop_bias]).to_csv(out_dir / 'sasrec_popularity_bias.csv', index=False)

    gender_df = demographic_bias(
        sasrec_recs, val_df, user_meta, 'gender_enc', {0: 'Female', 1: 'Male'},
    )
    logger.info(f'SASRec gender bias:\n{gender_df.to_string()}')
    gender_df.to_csv(out_dir / 'sasrec_gender_bias.csv', index=False)

    age_df = demographic_bias(sasrec_recs, val_df, user_meta, 'age_enc', AGE_LABELS)
    logger.info(f'SASRec age bias:\n{age_df.to_string()}')
    age_df.to_csv(out_dir / 'sasrec_age_bias.csv', index=False)

    occ_df = demographic_bias(sasrec_recs, val_df, user_meta, 'occupation', OCC_LABELS)
    occ_df.to_csv(out_dir / 'sasrec_occupation_bias.csv', index=False)

    genre_df = genre_concentration(sasrec_recs, genre_matrix)
    logger.info(f'SASRec genre concentration (top 5):\n{genre_df.head().to_string()}')
    genre_df.to_csv(out_dir / 'sasrec_genre_concentration.csv', index=False)

    # ---- SVD (popularity bias comparison only) ----
    logger.info('Computing SVD score matrix...')
    svd_scores = get_svd_score_matrix(artifacts)
    svd_scores = mask_seen(svd_scores, train_df)
    svd_recs = get_topk_recs(svd_scores, K=10)
    svd_pop = popularity_bias(svd_recs, item_pop, n_items)
    logger.info(f'SVD popularity ratio: {svd_pop["popularity_ratio"]:.3f}')
    pd.DataFrame([svd_pop]).to_csv(out_dir / 'svd_popularity_bias.csv', index=False)

    # ---- Summary ----
    summary = {
        'sasrec_popularity_ratio': pop_bias['popularity_ratio'],
        'sasrec_longtail_fraction': pop_bias['longtail_fraction'],
        'svd_popularity_ratio': svd_pop['popularity_ratio'],
        'svd_longtail_fraction': svd_pop['longtail_fraction'],
        'gender_gap_HR10': float(
            gender_df.set_index('group').loc['Male', 'HR@10']
            - gender_df.set_index('group').loc['Female', 'HR@10']
        ),
        'age_hr10_max': float(age_df['HR@10'].max()),
        'age_hr10_min': float(age_df['HR@10'].min()),
        'age_hr10_range': float(age_df['HR@10'].max() - age_df['HR@10'].min()),
        'top_genre': genre_df.iloc[0]['genre'],
        'top_genre_frac': float(genre_df.iloc[0]['fraction_in_recs']),
    }
    pd.DataFrame([summary]).to_csv(out_dir / 'bias_summary.csv', index=False)
    logger.info(f'Bias audit complete. Results saved to {out_dir}')
    for k, v in summary.items():
        logger.info(f'  {k}: {v}')
    return summary


if __name__ == '__main__':
    run()
