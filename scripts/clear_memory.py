"""
scripts/clear_memory.py
Run this before train_two_tower.py to free RAM and GPU memory
from previous sections (SVD ~14GB, CF matrices ~350MB).
Hard stops if RTX 3090 VRAM < 20GB free.
"""
import gc
import os
import sys
import psutil
import torch
from pathlib import Path


VRAM_MINIMUM_GB = 20.0  # hard stop threshold for RTX 3090
RAM_TARGET_GB = 10.0    # warn if RAM available drops below this after cleanup


def bytes_to_gb(b): return b / 1e9


def get_ram_usage():
    proc = psutil.Process(os.getpid())
    return bytes_to_gb(proc.memory_info().rss)


def get_system_ram():
    vm = psutil.virtual_memory()
    return bytes_to_gb(vm.used), bytes_to_gb(vm.available)


def get_gpu_stats():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    total = bytes_to_gb(props.total_memory)
    reserved = bytes_to_gb(torch.cuda.memory_reserved(0))
    allocated = bytes_to_gb(torch.cuda.memory_allocated(0))
    free = total - reserved
    return {'total': total, 'reserved': reserved, 'allocated': allocated, 'free': free}


def print_stats(label):
    ram_used, ram_avail = get_system_ram()
    print(f'\n[{label}]')
    print(f'  System RAM used: {ram_used:.1f} GB | available: {ram_avail:.1f} GB')
    gpu = get_gpu_stats()
    if gpu:
        print(f'  GPU VRAM total: {gpu["total"]:.1f} GB | reserved: {gpu["reserved"]:.1f} GB | free: {gpu["free"]:.1f} GB')
    else:
        print('  GPU: not available')


def clear_gpu():
    if not torch.cuda.is_available():
        print('No GPU found — skipping GPU cleanup.')
        return
    print('Clearing GPU cache...')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_accumulated_memory_stats(0)
    print('GPU cache cleared.')


def clear_ram():
    print('Running garbage collection...')
    for _ in range(3):
        gc.collect()
    print('GC complete.')


def kill_large_objects():
    """Note any heavy modules still loaded; full unload requires Python restart."""
    killed = []
    for mod_name in ['implicit', 'sentence_transformers']:
        if mod_name in sys.modules:
            killed.append(mod_name)
    if killed:
        print(f'Heavy modules still loaded: {killed} — consider restarting Python if RAM is still high')


def verify_resources():
    """Hard stop if VRAM < threshold. Warn if RAM is low."""
    gpu = get_gpu_stats()
    if gpu:
        if gpu['free'] < VRAM_MINIMUM_GB:
            print(f'\nFATAL: Only {gpu["free"]:.1f} GB VRAM free. Need {VRAM_MINIMUM_GB} GB.')
            print('Run: nvidia-smi to check what is using GPU memory.')
            print('If another process is using GPU, kill it and re-run this script.')
            sys.exit(1)
        else:
            print(f'GPU OK: {gpu["free"]:.1f} GB VRAM free (need {VRAM_MINIMUM_GB} GB)')
    _, ram_avail = get_system_ram()
    if ram_avail < RAM_TARGET_GB:
        print(f'WARNING: Only {ram_avail:.1f} GB RAM available. Training may be slow.')
        print('Consider restarting the Python session to fully free RAM from previous sections.')
    else:
        print(f'RAM OK: {ram_avail:.1f} GB available')


def run():
    print('=' * 60)
    print('Serko RecSys — Pre-Training Memory Cleanup')
    print('Target: RTX 3090 (24GB VRAM), 64GB host RAM')
    print('=' * 60)

    print_stats('BEFORE CLEANUP')
    clear_gpu()
    kill_large_objects()
    clear_ram()
    print_stats('AFTER CLEANUP')
    verify_resources()

    print('\nMemory cleanup complete. Safe to run train_two_tower.py.')


if __name__ == '__main__':
    run()
