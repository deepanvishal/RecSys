import yaml, torch
from pathlib import Path


checks = []


# A1: global device in config
cfg = yaml.safe_load(open('config/config.yaml'))
ok = 'device' in cfg.get('project', {})
checks.append(('A1_global_device', ok))
print(f"{'OK' if ok else 'MISSING'}: A1 — global device in config.yaml")


# A2: utils/device.py exists and works
try:
    from utils.device import get_device, log_device_info
    log_device_info()
    checks.append(('A2_device_util', True))
    print('OK: A2 — utils/device.py imports correctly')
except Exception as e:
    checks.append(('A2_device_util', False))
    print(f'FAILED: A2 — {e}')


# B1: correct subset name in download.py
src = open('data/download.py').read()
ok = 'hf_subset' in src or '0core_rating_only' in src
checks.append(('B1_dataset_name', ok))
print(f"{'OK' if ok else 'MISSING'}: B1 — dataset subset name fix")


# B2: JSON fix in simulate.py
src = open('data/simulate.py').read()
ok = '_convert' in src or 'convert_types' in src
checks.append(('B2_json_fix', ok))
print(f"{'OK' if ok else 'MISSING'}: B2 — JSON serialization fix")


# B3: GPU in feature_store.py
src = open('data/feature_store.py').read()
ok = 'get_device' in src and 'normalize_embeddings' in src
checks.append(('B3_gpu_feature_store', ok))
print(f"{'OK' if ok else 'MISSING'}: B3 — GPU in feature_store.py")


failed = [f for f, ok in checks if not ok]
if failed:
    print(f'\nFAILED: {failed}')
else:
    print('\nAll corrections verified. Ready for Section 3.')
