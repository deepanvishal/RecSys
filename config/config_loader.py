import yaml
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


_config = None


def load_config(config_path: str = 'config/config.yaml') -> dict:
    global _config
    if _config is None:
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config


def get_config() -> dict:
    if _config is None:
        return load_config()
    return _config


def get_wandb_config() -> dict:
    cfg = get_config()
    return {
        'project': os.getenv('WANDB_PROJECT', cfg['wandb']['project']),
        'entity': os.getenv('WANDB_ENTITY', cfg['wandb']['entity']),
        'api_key': os.getenv('WANDB_API_KEY'),
    }
