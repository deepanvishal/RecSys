from pathlib import Path
import numpy as np
import torch


checks = []


# Model weights saved
ok = Path('artifacts/models/two_tower/best_model.pt').exists()
checks.append(('model_saved', ok))
print(f"{'OK' if ok else 'MISSING'}: two_tower best_model.pt")


# FAISS index saved
ok = Path('artifacts/indexes/two_tower.index').exists()
checks.append(('faiss_index', ok))
print(f"{'OK' if ok else 'MISSING'}: FAISS index")


# Leakage analysis saved
ok = Path('report/leakage_analysis.json').exists()
checks.append(('leakage_analysis', ok))
print(f"{'OK' if ok else 'MISSING'}: leakage_analysis.json")


# Model loads and runs forward pass
try:
    from models.two_tower import TwoTowerModel
    from utils.device import get_device
    device = get_device()
    model = TwoTowerModel()
    model.load_state_dict(torch.load('artifacts/models/two_tower/best_model.pt', map_location=device))
    model.to(device).eval()
    dummy_user = torch.randn(4, 384).to(device)
    user_vecs = model.encode_user(dummy_user)
    assert user_vecs.shape == (4, 128), f'Expected (4,128), got {user_vecs.shape}'
    dummy_item = torch.randn(4, 384).to(device)
    dummy_price = torch.zeros(4, dtype=torch.long).to(device)
    dummy_pop = torch.rand(4).to(device)
    item_vecs = model.encode_items(dummy_item, dummy_price, dummy_pop)
    assert item_vecs.shape == (4, 128)
    checks.append(('model_forward', True))
    print('OK: model loads, user/item encoding works')
except Exception as e:
    checks.append(('model_forward', False))
    print(f'FAILED: model forward — {e}')


# FAISS retrieval works
try:
    from inference.retrieval import FAISSRetriever
    r = FAISSRetriever()
    r.load()
    dummy_q = np.random.randn(2, 128).astype(np.float32)
    scores, indices = r.search(dummy_q, k=10)
    assert indices.shape == (2, 10)
    checks.append(('faiss_search', True))
    print('OK: FAISS retrieval works')
except Exception as e:
    checks.append(('faiss_search', False))
    print(f'FAILED: FAISS — {e}')


failed = [f for f, ok in checks if not ok]
if failed:
    print(f'FAILED: {failed}')
else:
    print('Section 5 complete. All checks passed. Proceed to Section 6.')
