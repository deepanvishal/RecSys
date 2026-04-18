# Serko RecSys

Production-grade Recommender System built for the Serko Senior Data Scientist take-home assignment. Targets the Amazon Reviews 2023 (Electronics) dataset and demonstrates a tiered architecture (trending → SVD → two-tower) suitable for cold, sparse, and warm users.

## Project Layout

```
RecSys/
├── config/           # config.yaml + config_loader (single source of truth)
├── data/             # download, preprocess, simulate, feature_store
├── models/           # CF, SVD, two-tower implementations
├── training/         # training entrypoints with W&B logging
├── inference/        # tiered router, reranker, incremental update
├── evaluation/       # metrics, bias audit, cohort analysis
├── api/              # FastAPI service
├── ui/               # Streamlit demo
├── notebooks/        # EDA, comparison, bias notebooks
├── tests/            # pytest unit + integration tests
├── utils/            # logger, seed
├── artifacts/        # models, embeddings, FAISS indexes, cache (gitignored)
├── logs/             # runtime logs (gitignored)
└── report/           # technical_report.md
```

## Setup

1. Create and activate a virtualenv (Python 3.10+).
2. Install dependencies: `pip install -r requirements.txt`
3. Install package in editable mode: `pip install -e .`
4. Copy `.env.example` to `.env` and fill in `WANDB_API_KEY` / `WANDB_ENTITY`.

## Configuration Contract

- All hyperparameters and paths live in [config/config.yaml](config/config.yaml).
- All modules read config via [config/config_loader.py](config/config_loader.py): `from config.config_loader import get_config`.
- W&B credentials come from environment variables — never hardcoded.
- All paths are relative to repo root.

## Logging & Reproducibility

- All modules log via [utils/logger.py](utils/logger.py): `from utils.logger import get_logger`.
- All training scripts call `set_seed()` from [utils/seed.py](utils/seed.py) immediately after imports.

## Build Sections

| Section | Scope | Status |
| --- | --- | --- |
| 1 — Foundation | Structure, config, utils | Complete |
| 2 — Data Pipeline | Download, preprocess, simulate, features | Pending |
| 3 — Model: CF | Collaborative filtering + W&B | Pending |
| 4 — Model: SVD | Matrix factorization + W&B | Pending |
| 5 — Model: Two-Tower | PyTorch neural + FAISS + W&B | Pending |
| 6 — Inference Engine | Tiered router, reranker, incremental update | Pending |
| 7 — Evaluation | Metrics, bias audit, cohort analysis | Pending |
| 8 — API | FastAPI endpoints + tests | Pending |
| 9 — UI | Streamlit 5-panel demo | Pending |
| 10 — Notebooks + Report | EDA, comparison, bias, report | Pending |

## Validation

After completing each section, run the section's verification script. For Section 1:

```
python verify_structure.py
```
