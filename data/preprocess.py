import pandas as pd
import numpy as np
import json
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('preprocess')


GENRES = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


def _genre_col(g):
    return 'genre_' + g.lower().replace('-', '_').replace("'", '')


def load_ratings(ml_dir, sep):
    df = pd.read_csv(
        ml_dir / 'ratings.dat', sep=sep, engine='python',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1',
    )
    df['user_id'] = df['user_id'].astype('int32') - 1   # 0-indexed
    df['item_id'] = df['item_id'].astype('int32') - 1
    df['rating'] = df['rating'].astype('float32')
    df['implicit_feedback'] = 1
    logger.info(f'Ratings: {df.shape}, users: {df.user_id.nunique()}, items: {df.item_id.nunique()}')
    return df


def load_movies(ml_dir, sep):
    movies = pd.read_csv(
        ml_dir / 'movies.dat', sep=sep, engine='python',
        names=['item_id', 'title', 'genres'],
        encoding='latin-1',
    )
    movies['item_id'] = movies['item_id'].astype('int32') - 1
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float32')
    for g in GENRES:
        movies[_genre_col(g)] = movies['genres'].str.contains(g, regex=False).astype('int8')
    movies['genre_vector'] = movies['genres'].apply(
        lambda x: [int(g in x) for g in GENRES]
    )
    logger.info(f'Movies: {movies.shape}')
    return movies


def load_users(ml_dir, sep):
    users = pd.read_csv(
        ml_dir / 'users.dat', sep=sep, engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip'],
        encoding='latin-1',
    )
    users['user_id'] = users['user_id'].astype('int32') - 1
    users['gender_enc'] = (users['gender'] == 'M').astype('int8')
    age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    users['age_enc'] = users['age'].map(age_map).fillna(0).astype('int8')
    users['occupation'] = users['occupation'].astype('int8')
    logger.info(f'Users: {users.shape}')
    return users


def leave_one_out_split(df):
    logger.info('Applying leave-one-out split (standard ML-1M benchmark)...')
    df = df.sort_values(['user_id', 'timestamp'])
    df['rank'] = df.groupby('user_id').cumcount(ascending=False)
    df['split'] = 'train'
    df.loc[df['rank'] == 0, 'split'] = 'test'
    df.loc[df['rank'] == 1, 'split'] = 'val'
    df = df.drop(columns=['rank'])
    logger.info(f'  Train: {(df.split=="train").sum()}')
    logger.info(f'  Val:   {(df.split=="val").sum()}')
    logger.info(f'  Test:  {(df.split=="test").sum()}')
    return df


def run():
    cfg = get_config()
    ml_dir = Path(cfg['data']['raw_dir']) / 'ml-1m'
    sep = cfg['data']['delimiter']
    out = Path(cfg['data']['processed_dir'])
    out.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(ml_dir, sep)
    movies = load_movies(ml_dir, sep)
    users = load_users(ml_dir, sep)

    df = ratings.merge(
        movies[['item_id', 'title', 'year', 'genres', 'genre_vector']],
        on='item_id', how='left',
    )
    df = df.merge(
        users[['user_id', 'gender', 'gender_enc', 'age', 'age_enc', 'occupation']],
        on='user_id', how='left',
    )

    item_pop = df.groupby('item_id').size().rename('item_pop')
    df = df.merge(item_pop, on='item_id')
    df['popularity_rank'] = (df['item_pop'] / df['item_pop'].max()).astype('float32')
    df = df.drop(columns=['item_pop'])

    df = leave_one_out_split(df)

    for split in ['train', 'val', 'test']:
        sub = df[df['split'] == split].drop(columns=['split'])
        sub.to_parquet(out / f'{split}.parquet', index=False)
    df.to_parquet(out / 'full.parquet', index=False)

    item_meta = movies[['item_id', 'title', 'year', 'genres', 'genre_vector']].copy()
    item_meta.to_parquet(out / 'item_metadata.parquet', index=False)

    user_meta = users[['user_id', 'gender', 'gender_enc', 'age', 'age_enc', 'occupation']].copy()
    user_meta.to_parquet(out / 'user_metadata.parquet', index=False)

    n_users = int(df['user_id'].max()) + 1
    n_items = int(df['item_id'].max()) + 1
    n_train = int((df['split'] == 'train').sum())
    with open(out / 'dataset_stats.json', 'w') as f:
        json.dump({
            'n_users': n_users, 'n_items': n_items,
            'n_train': n_train,
            'n_genres': len(GENRES), 'genres': GENRES,
        }, f, indent=2)

    logger.info(f'Preprocessing complete. n_users={n_users}, n_items={n_items}')
    return df


if __name__ == '__main__':
    run()
