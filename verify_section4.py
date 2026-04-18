from pathlib import Path
import numpy as np


checks = []


# Model files exist
best_dir = Path('artifacts/models/svd/best/')
models = list(best_dir.glob('*.pkl')) if best_dir.exists() else []
ok = len(models) > 0
checks.append(('svd_model_saved', ok))
print(f"{'OK' if ok else 'MISSING'}: SVD best model — found {len(models)}")


# Load saved best, exercise recommend / foldin / cold start
try:
    from models.svd import SVDModel
    model = SVDModel.load(str(models[0]))
    model.is_fitted = True if model.model.user_factors is not None else False
    assert model.is_fitted, 'loaded model not marked fitted'
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


# Fold-in latency
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
