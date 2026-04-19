from data.download import download_movielens
from data.preprocess import run as preprocess
from data.simulate import run as simulate
from data.feature_store import run as build_features
from utils.logger import get_logger


logger = get_logger('run_pipeline')


if __name__ == '__main__':
    logger.info('=== Section 2: Data Pipeline ===')
    logger.info('Step 1/4: Download')
    download_movielens()
    logger.info('Step 2/4: Preprocess')
    preprocess()
    logger.info('Step 3/4: Simulate')
    simulate()
    logger.info('Step 4/4: Feature Store')
    build_features()
    logger.info('=== Data Pipeline Complete ===')
