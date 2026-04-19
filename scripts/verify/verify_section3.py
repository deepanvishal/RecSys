from pathlib import Path
import joblib
import numpy as np


checks = []


# Check at least one best CF model exists
best_dir = Path('artifacts/models/cf/best/')
models_found = list(best_dir.glob('*.pkl')) if best_dir.exists() else []
ok = len(models_found) > 0
checks.append(('cf_models_saved', ok))
print(f"{'OK' if ok else 'MISSING'}: CF best models — found {len(models_found)}")


# Check metrics module
try:
    from evaluation.metrics import hit_at_k, ndcg_at_k, catalog_coverage
    h = hit_at_k(5, [1, 2, 5, 3], 5)
    assert h == 1.0
    h2 = hit_at_k(9, [1, 2, 5, 3], 5)
    assert h2 == 0.0
    checks.append(('metrics_module', True))
    print('OK: metrics module imports and unit tests pass')
except Exception as e:
    checks.append(('metrics_module', False))
    print(f'FAILED: metrics module — {e}')


# Check CF model loads and recommends
try:
    import scipy.sparse as sp
    from models.cf import CollaborativeFilter
    from config.config_loader import get_config
    cfg = get_config()
    matrix = sp.load_npz('artifacts/cache/cf_interaction_matrix.npz')
    model = CollaborativeFilter(model_type='item', k=10)
    model.fit(matrix)
    # Find a warm user (any with nnz > 0) for recommend test
    nnz_per_user = np.diff(matrix.indptr)
    warm_user = int(np.argmax(nnz_per_user))
    recs = model.recommend(warm_user, n=10)
    assert len(recs) == 10
    cold_recs = model._popularity_fallback(10, [])
    assert len(cold_recs) == 10
    checks.append(('cf_recommend', True))
    print('OK: CF recommend works for warm and cold users')
except Exception as e:
    checks.append(('cf_recommend', False))
    print(f'FAILED: CF recommend — {e}')


failed = [f for f, ok in checks if not ok]
if failed:
    print(f'FAILED checks: {failed}')
else:
    print('Section 3 complete. All checks passed. Proceed to Section 4.')
