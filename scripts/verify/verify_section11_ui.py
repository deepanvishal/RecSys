"""
Headless verifier for Section 11 — confirms the Streamlit app's data
dependencies + the live API are reachable. Does not launch the browser UI.
"""
from pathlib import Path
import importlib
import pandas as pd
import requests


checks = []

# UI script + module
ui_path = Path('ui/app.py')
checks.append(('ui_app_exists', ui_path.exists()))

st_ok = False
try:
    importlib.import_module('streamlit')
    st_ok = True
except Exception as e:
    print(f'streamlit import failed: {e}')
checks.append(('streamlit_import', st_ok))

# Data dependencies
movies_path = Path('data/processed/item_metadata.parquet')
movies_ok = False
if movies_path.exists():
    try:
        m = pd.read_parquet(movies_path).set_index('item_id')[['title', 'genres']]
        movies_ok = len(m) > 0
    except Exception as e:
        print(f'movies load failed: {e}')
checks.append(('movies_loadable', movies_ok))

# Bias CSVs
bias_dir = Path('artifacts/bias')
required_csvs = [
    'bias_summary.csv',
    'sasrec_gender_bias.csv',
    'sasrec_age_bias.csv',
    'sasrec_genre_concentration.csv',
    'sasrec_popularity_bias.csv',
    'svd_popularity_bias.csv',
]
csv_ok = all((bias_dir / c).exists() for c in required_csvs)
checks.append(('bias_csvs_present', csv_ok))

# Live API
api_ok = False
api_health = None
try:
    r = requests.get('http://localhost:8000/health', timeout=3)
    api_ok = r.status_code == 200
    api_health = r.json() if api_ok else None
except Exception as e:
    print(f'API unreachable: {e}')
checks.append(('api_reachable', api_ok))

# End-to-end smoke: hit /recommend through the same path the UI uses
e2e_ok = False
if api_ok:
    try:
        r = requests.post(
            'http://localhost:8000/recommend',
            json={'history': [50, 296, 527, 858], 'n': 5}, timeout=10,
        )
        body = r.json()
        e2e_ok = r.status_code == 200 and len(body.get('items', [])) == 5
    except Exception as e:
        print(f'E2E /recommend failed: {e}')
checks.append(('api_recommend_e2e', e2e_ok))


for name, ok in checks:
    print(f"{'OK' if ok else 'FAIL'}: {name}")

failed = [n for n, ok in checks if not ok]
if api_health:
    print(f'\nAPI health payload: {api_health}')
if failed:
    print(f'\nFAILED: {failed}')
else:
    print('\nSection 11 UI prerequisites verified. Launch with: streamlit run ui/app.py')
