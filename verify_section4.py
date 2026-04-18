from pathlib import Path
import scipy.sparse as sp
import numpy as np


checks = []


# Model files exist
best_dir = Path('artifacts/models/svd/best/')
models = list(best_dir.glob('*.pkl')) if best_dir.exists() else []
ok = len(models) > 0
checks.append(('svd_model_saved', ok))
print(f"{'OK' if ok else 'MISSING'}: SVD best model — found {len(models)}")


# Model loads, fits, recommends
try:
    from models.svd import SVDModel
    from config.config_loader import get_config
    cfg = get_config()
    matrix = sp.load_npz('artifacts/cache/svd_item_user_matrix.npz')
    model = SVDModel(factors=64)
    model.fit(matrix)
    recs = model.recommend(0, n=10)
    assert len(recs) == 10, f'Expected 10 recs, got {len(recs)}'
    foldin_recs = model.foldin_recommend([0, 1, 2, 3], n=10)
    assert len(foldin_recs) == 10
    cold_recs = model.foldin_recommend([], n=10)
    assert len(cold_recs) == 10
    checks.append(('svd_recommend', True))
    print('OK: SVD recommend, fold-in, and cold start all work')
except Exception as e:
    checks.append(('svd_recommend', False))
    print(f'FAILED: SVD — {e}')


# Fold-in latency check
try:
    import time
    items = list(range(5))
    times = []
    for _ in range(100):
        t0 = time.time()
        model.foldin_recommend(items, n=10)
        times.append(time.time() - t0)
    avg_ms = np.mean(times) * 1000
    print(f'OK: fold-in avg latency = {avg_ms:.2f}ms per user')
    checks.append(('foldin_latency', True))
except Exception as e:
    checks.append(('foldin_latency', False))
    print(f'FAILED: fold-in latency — {e}')


failed = [f for f, ok in checks if not ok]
if failed:
    print(f'FAILED: {failed}')
else:
    print('Section 4 complete. All checks passed. Proceed to Section 5.')
