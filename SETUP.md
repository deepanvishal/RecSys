# Setup — Recreate Serko RecSys end-to-end

This guide takes a data scientist from a fresh machine to a working
deployment. Total time on a CUDA-equipped workstation: **~45 min** (most of
it training). On CPU-only: **~3 hours**.

## 0. Prerequisites

| Need | Why | How |
|---|---|---|
| Python 3.10 | Pinned for `implicit`, `torch 2.3` ABI | `pyenv install 3.10.13` or system |
| Git + Git LFS | Model artifacts > 10MB | `git lfs install` once |
| ~5 GB free disk | Raw data, processed parquets, model artifacts | — |
| (Optional) CUDA 12.1 + 8 GB GPU | SASRec/BERT4Rec training | Falls back to CPU automatically |
| (Optional) W&B account | Experiment tracking | `wandb login` — free tier is sufficient |

## 1. Clone & install

```bash
git clone https://github.com/deepanvishal/RecSys
cd RecSys

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

If you only need the trained models + serving (no retraining), skip to
[§5 Serving](#5-serving) — model artifacts are committed via Git LFS.

## 2. Data pipeline

```bash
# Download MovieLens 1M (~6 MB) → data/raw/ml-1m/
python data/download.py

# Build train/val/test parquets, item & user metadata, dataset stats
python data/preprocess.py

# Generate cohort splits (cold/sparse/warm) for cohort-level eval
python data/simulate.py

# Build feature store (item popularity, user genre preferences, etc.)
python data/feature_store.py
```

Outputs to `data/processed/`:
- `train.parquet`, `val.parquet`, `test.parquet`, `full.parquet`
- `item_metadata.parquet`, `user_metadata.parquet`
- `dataset_stats.json`
- `cohorts/user_cohorts.parquet`, `simulation/daily_snapshots.json`

## 3. Train models

Each script writes a single best checkpoint to `artifacts/models/<model>/`.

```bash
# Item-Item CF — 5s on CPU
python models/cf/train_cf.py

# ALS / SVD — 30s on CPU
python models/svd/train_svd.py

# Two-Tower — ~5 min on RTX 3090, ~30 min on CPU
python models/two_tower/train_two_tower.py

# SASRec — ~7 min on RTX 3090, ~1 hour on CPU
python models/sasrec/train_sasrec.py

# BERT4Rec — ~7 min on RTX 3090, similar caveat
python models/bert4rec/train_bert4rec.py
```

Per-script outputs:
- `artifacts/models/cf/item_item_best.pkl`
- `artifacts/models/svd/als_best.pkl`
- `artifacts/models/two_tower/two_tower_best.pt`
- `artifacts/models/sasrec/sasrec_best.pt`
- `artifacts/models/bert4rec/bert4rec_best.pt`

W&B runs land under project `serko-recsys-movielens` if you ran `wandb login`.
Disable with `WANDB_MODE=disabled python models/sasrec/train_sasrec.py`.

## 4. Build inference assets

```bash
# Pre-compute Two-Tower embeddings + FAISS index (item_emb, user_emb, faiss.index)
python inference/precompute_embeddings.py

# Generate reranker training data from CF + SVD + Two-Tower top-50 candidates
python inference/reranker_data.py

# Train LightGBM reranker (val AUC ~0.74)
python inference/reranker.py

# Run bias audit (SASRec + reranked)
python evaluation/bias_audit.py
python evaluation/bias_audit.py reranked

# (Optional) latency benchmark on the tiered engine
python inference/latency_benchmark.py
```

Outputs to `artifacts/`:
- `two_tower/{item_emb,user_emb,faiss.index,item_content_emb,item_genre_matrix}.npy`
- `reranker/{reranker.pkl,feature_importance.csv,reranker_train_*.parquet}`
- `bias/{bias_summary,sasrec_*,reranked_*,svd_*}.csv`

## 5. Serving

### Local (two terminals)

```bash
# Terminal 1 — FastAPI
python -m api.main
# → http://localhost:8000/docs

# Terminal 2 — Streamlit UI
streamlit run ui/app.py
# → http://localhost:8501
```

### Docker (single container, both processes)

```bash
docker build -t serko-recsys .
docker run -p 7860:7860 serko-recsys
# → http://localhost:7860
```

### Hugging Face Spaces

See [DEPLOY.md](DEPLOY.md) for the push-to-Spaces workflow.

## 6. Verifying the install

```bash
# Section-by-section verification scripts (one per build phase)
python scripts/verify/check_ui_ready.py        # Final readiness check
python scripts/verify/verify_section8_engine.py  # Engine sanity
python scripts/verify/verify_section11_ui.py     # UI artifacts
```

These scripts assert that artifacts exist with expected shapes and that all
API endpoints respond — useful as a `make smoke-test` substitute.

## 7. Configuration

All hyperparameters and paths live in `config/config.yaml`. Read via
`config.config_loader.get_config()`. Override at runtime with the
`SERKO_CONFIG` env var pointing to a different YAML.

Key sections:
- `data` — dataset paths, train/val/test fractions
- `models.<model_name>` — per-model hyperparameters
- `serving` — API port, engine cache sizes
- `wandb` — project, entity (set `mode: disabled` for offline)

## 8. Tests

```bash
pytest tests/ -v
```

Covers metrics correctness, tiered engine routing, and API contract.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: implicit` | Wrong Python version | `pyenv local 3.10.13` |
| SASRec NaN loss after 1 epoch | Combined causal + key padding mask | Use only causal mask (already fixed in `models/sasrec/model.py`) |
| FAISS GPU crash on torch 12.1 | `faiss-gpu` ABI mismatch | Use `faiss-cpu` (`requirements.docker.txt` is correct) |
| Streamlit "API not reachable" | API not started, or wrong port | Check `python -m api.main` is running |
| `WandbAuthError` during training | Not logged in | `wandb login` or set `WANDB_MODE=disabled` |
| Docker push to HF rejects binaries | Files committed pre-LFS | Run `git lfs migrate import --everything --include="*.pkl,*.pt,*.npy,*.parquet,*.index"` |

## 10. Recompute everything

```bash
# Nuke all outputs
rm -rf data/raw data/processed artifacts wandb logs

# Then re-run §2 through §4
```
