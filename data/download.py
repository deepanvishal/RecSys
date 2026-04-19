import zipfile
import urllib.request
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('download')


def download_movielens():
    cfg = get_config()
    raw_dir = Path(cfg['data']['raw_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / 'ml-1m.zip'
    ml_dir = raw_dir / 'ml-1m'

    if ml_dir.exists() and (ml_dir / 'ratings.dat').exists():
        logger.info('MovieLens 1M already downloaded.')
        return ml_dir

    logger.info(f'Downloading from {cfg["data"]["download_url"]}...')
    urllib.request.urlretrieve(cfg['data']['download_url'], zip_path)
    logger.info(f'Downloaded {zip_path.stat().st_size / 1e6:.1f} MB')

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(raw_dir)
    zip_path.unlink()

    required = ['ratings.dat', 'movies.dat', 'users.dat']
    for f in required:
        assert (ml_dir / f).exists(), f'Missing {f}'
    logger.info(f'MovieLens 1M extracted to {ml_dir}')
    return ml_dir


if __name__ == '__main__':
    download_movielens()
