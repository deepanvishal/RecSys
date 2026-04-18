import random
import numpy as np
import torch
from config.config_loader import get_config


def set_seed(seed: int = None):
    if seed is None:
        seed = get_config()['project']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
