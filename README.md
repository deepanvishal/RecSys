---
title: Serko RecSys
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Serko RecSys — MovieLens 1M

Production-grade recommender system built as a take-home assignment.
Five models (CF, SVD, Two-Tower, SASRec, BERT4Rec) on MovieLens 1M,
tiered inference engine, bias audit, FastAPI, and Streamlit UI.

## Results

| Model | HR@10 | NDCG@10 | Role |
|---|---|---|---|
| SASRec | **0.3328** | **0.1873** | Tier 3 — warm users |
| SVD (f=128) | 0.1020 | 0.0487 | Tier 2 — sparse users |
| UserUserCF | 0.0830 | 0.0404 | Tier 2 — fallback |
| ItemItemCF | 0.0816 | 0.0390 | Tier 2 — item similarity |
| Two-Tower | 0.0647 | 0.0315 | Cold-item retrieval |
| BERT4Rec | 0.0417 | 0.0199 | Experimental (needs 200+ epochs) |

All metrics: full-catalog evaluation (3,952 items), leave-one-out split.

## Architecture

```
User request
    │
    ├── 0 interactions  → Tier 1: Popularity
    ├── 1–5 interactions → Tier 2: SVD fold-in
    └── >5 interactions  → Tier 3: SASRec

New/unseen items → Two-Tower content retrieval
```

## Setup

```bash
git clone https://github.com/deepanvishal/RecSys
cd RecSys
pip install -r requirements.txt
```

## Run the full pipeline

```bash
# Download + preprocess data
python data/download.py
python data/preprocess.py
python data/simulate.py
python data/feature_store.py

# Train all models
python models/cf/train_cf.py
python models/svd/train_svd.py
python models/two_tower/train_two_tower.py
python models/sasrec/train_sasrec.py
python models/bert4rec/train_bert4rec.py

# Bias audit
python evaluation/bias_audit.py

# Serve
python -m api.main          # FastAPI on :8000
streamlit run ui/app.py     # UI on :8501
```

## W&B Dashboard

https://wandb.ai/deepanvishal-cvs-health/serko-recsys-movielens

## Dataset

MovieLens 1M (GroupLens). 6,040 users, 3,706 movies, 1,000,209 ratings.
Downloaded automatically by data/download.py.

## Project structure

```
config/          config.yaml + loader
data/            download, preprocess, simulate, feature_store
models/          cf, svd, two_tower, sasrec, bert4rec
evaluation/      metrics.py, bias_audit.py
inference/       engine.py, latency_benchmark.py, cohort_eval.py
api/             main.py (FastAPI)
ui/              app.py (Streamlit)
artifacts/       trained models, embeddings, bias CSVs
```
