import torch
from config.config_loader import get_config


def get_device() -> torch.device:
    cfg = get_config()
    requested = cfg['project'].get('device', 'cuda')
    if requested == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_device_name() -> str:
    device = get_device()
    if device.type == 'cuda':
        return torch.cuda.get_device_name(0)
    return 'CPU'


def log_device_info():
    device = get_device()
    if device.type == 'cuda':
        print(f'Device: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        print(f'CUDA: {torch.version.cuda}')
    else:
        print('Device: CPU')
