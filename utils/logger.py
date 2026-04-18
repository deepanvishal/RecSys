import logging
import os
from pathlib import Path
from config.config_loader import get_config


def get_logger(name: str) -> logging.Logger:
    cfg = get_config()
    Path(cfg['paths']['logs']).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fh = logging.FileHandler(f"{cfg['paths']['logs']}/{name}.log")
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger
