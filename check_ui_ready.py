"""
Pre-UI readiness check.
Run from project root: python check_ui_ready.py
"""
import sys
from pathlib import Path

BASE = 'http://localhost:8000'
failures = []
warnings = []

def ok(msg):  print(f"  OK  {msg}")
def fail(msg): print(f"  FAIL {msg}"); failures.append(msg)
def warn(msg): print(f"  WARN {msg}"); warnings.append(msg)

# Artifacts
print("\n[1] Artifacts")

required_files = {
    'artifacts/models/cf/item_item_best.pkl':           'CF model',
    'artifacts/models/svd/als_best.pkl':                'SVD model',
    'artifacts/models/sasrec/sasrec_best.pt':           'SASRec model',
    'artifacts/models/bert4rec/bert4rec_best.pt':       'BERT4Rec model',
    'artifacts/models/two_tower/two_tower_best.pt':     'Two-Tower model',
    'artifacts/two_tower/item_content_emb.npy':         'Item content embeddings',
    'artifacts/two_tower/item_genre_matrix.npy':        'Item genre matrix',
    'artifacts/two_tower/item_emb.npy':                 'Two-Tower item embeddings',
    'artifacts/two_tower/user_emb.npy':                 'Two-Tower user embeddings',
    'artifacts/two_tower/faiss.index':                  'FAISS index',
    'artifacts/reranker/reranker.pkl':                  'Reranker model',
    'artifacts/reranker/feature_importance.csv':        'Feature importance',
    'artifacts/bias/bias_summary.csv':                  'Bias summary',
    'artifacts/bias/sasrec_popularity_bias.csv':        'SASRec popularity bias',
    'artifacts/bias/sasrec_gender_bias.csv':            'SASRec gender bias',
    'artifacts/bias/sasrec_age_bias.csv':               'SASRec age bias',
    'artifacts/bias/sasrec_genre_concentration.csv':    'SASRec genre concentration',
    'artifacts/bias/svd_popularity_bias.csv':           'SVD popularity bias',
    'artifacts/bias/reranked_popularity_bias.csv':      'Reranked popularity bias',
    'artifacts/bias/reranked_gender_bias.csv':          'Reranked gender bias',
    'artifacts/bias/reranked_age_bias.csv':             'Reranked age bias',
    'artifacts/bias/reranked_genre_concentration.csv':  'Reranked genre concentration',
    'data/processed/train.parquet':                     'Train data',
    'data/processed/val.parquet':                       'Val data',
    'data/processed/item_metadata.parquet':             'Item metadata',
    'data/processed/user_metadata.parquet':             'User metadata',
    'data/processed/dataset_stats.json':                'Dataset stats',
}

for path, label in required_files.items():
    if Path(path).exists():
        ok(f"{label} ({path})")
    else:
        fail(f"MISSING -- {label} ({path})")

# bias_summary.csv keys
print("\n[2] bias_summary.csv keys")
try:
    import pandas as pd
    summary = pd.read_csv('artifacts/bias/bias_summary.csv').iloc[0]
    required_keys = [
        'sasrec_popularity_ratio', 'sasrec_longtail_fraction',
        'svd_popularity_ratio', 'gender_gap_HR10', 'age_hr10_range',
        'reranked_popularity_ratio', 'reranked_gender_gap_HR10',
    ]
    for k in required_keys:
        if k in summary.index:
            ok(f"{k} = {summary[k]:.4f}")
        else:
            fail(f"Missing key in bias_summary: {k}")
except Exception as e:
    fail(f"Could not read bias_summary.csv: {e}")

# API endpoints
print("\n[3] API endpoints")
try:
    import requests
    history = [50, 296, 527, 858, 1197, 1240]

    r = requests.get(f'{BASE}/health', timeout=5)
    if r.status_code == 200:
        caps = r.json().get('capabilities', [])
        ok(f"/health -- capabilities: {caps}")
    else:
        fail(f"/health returned {r.status_code}")

    r = requests.post(f'{BASE}/recommend',
                      json={'history': history, 'n': 10}, timeout=10)
    if r.status_code == 200:
        ok(f"/recommend -- tier={r.json()['tier']}, model={r.json()['model']}")
    else:
        fail(f"/recommend returned {r.status_code}")

    r = requests.post(f'{BASE}/recommend_all',
                      json={'history': history, 'n': 10}, timeout=15)
    if r.status_code == 200:
        data = r.json()
        keys = [k for k in ['cf','svd','two_tower','sasrec','bert4rec','reranked'] if k in data]
        ok(f"/recommend_all -- keys present: {keys}")
        if len(keys) < 6:
            fail(f"/recommend_all missing keys: {[k for k in ['cf','svd','two_tower','sasrec','bert4rec','reranked'] if k not in data]}")
    else:
        fail(f"/recommend_all returned {r.status_code}")

    r = requests.post(f'{BASE}/item_similarity',
                      json={'item_id': 50, 'n': 10}, timeout=10)
    if r.status_code == 200:
        models = list(r.json().keys())
        ok(f"/item_similarity -- models: {models}")
    else:
        fail(f"/item_similarity returned {r.status_code}")

    r = requests.post(f'{BASE}/similar_items',
                      json={'item_id': 50, 'n': 10}, timeout=10)
    if r.status_code == 200:
        ok(f"/similar_items -- {len(r.json()['items'])} items")
    else:
        fail(f"/similar_items returned {r.status_code}")

    r = requests.post(f'{BASE}/new_item_cold_start', json={
        'title': 'The Dark Knight (2008)',
        'genres': ['Action', 'Crime', 'Thriller'],
        'n_similar': 10, 'n_users': 20,
    }, timeout=15)
    if r.status_code == 200:
        data = r.json()
        ok(f"/new_item_cold_start -- {len(data['similar_items']['items'])} similar items, gender_bias present={('gender_bias' in data)}")
    else:
        fail(f"/new_item_cold_start returned {r.status_code}")

except requests.exceptions.ConnectionError:
    fail("API server not reachable on localhost:8000 -- start it first: python -m api.main")
except Exception as e:
    fail(f"API check error: {e}")

# Summary
print("\n" + "="*60)
if failures:
    print(f"NOT READY -- {len(failures)} failure(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
elif warnings:
    print(f"READY (with {len(warnings)} warning(s)) -- safe to build UI.")
else:
    print("READY -- all checks passed. Proceed to UI build (Step 4).")
