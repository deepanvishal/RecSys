# Serko RecSys — MovieLens 1M

A production-grade recommender system. Five models, a tiered inference engine,
a LightGBM reranker, full bias audit, FastAPI service, and a Netflix-style
Streamlit UI — all live in a single deployable container.

**🎬 Live demo:** <https://huggingface.co/spaces/deepanvishal/serko-recsys>
**📊 Training runs (W&B):** <https://wandb.ai/deepanvishal-cvs-health/serko-recsys-movielens>

---

## Headline results

| Model | HR@10 | NDCG@10 | Role in production |
|---|---|---|---|
| **SASRec** | **0.3328** | **0.1873** | Tier 3 — warm users (>5 interactions) |
| SVD (f=128) | 0.1020 | 0.0487 | Tier 2 — sparse users (1–5 interactions) |
| UserUserCF | 0.0830 | 0.0404 | Tier 2 — fallback |
| ItemItemCF | 0.0816 | 0.0390 | Item-similarity surface |
| Two-Tower | 0.0647 | 0.0315 | Cold-item retrieval, demographic-only cold-user |
| BERT4Rec | 0.0417 | 0.0199 | Experimental (under-trained, see report) |

All metrics: full-catalog evaluation (3,952 items), leave-one-out split.
Reranker validation AUC: 0.7417 (LightGBM, 21 features).

---

## What the demo shows — five tabs

### 1. For You — Personalised Homepage
Netflix-style row layout. Pick a user profile (Cold / Sparse / Warm / New User),
and the homepage rebuilds in real-time. Routing logic lives in the engine:

| User type | Rows shown | Models used |
|---|---|---|
| **Warm** (>5 interactions) | "Based on your watch history" + "Trending" + 2 genre rows + "New arrivals" | SASRec → genre-filtered Two-Tower → popularity → simulated |
| **Cold** (≤5 interactions) | "Trending" + 3 demographic-genre rows + "New arrivals" | popularity + demographic-conditioned Two-Tower |

### 2. Item Similarity — Compare three notions of "similar"
Pick a movie, see the top-N most similar items from three models stacked
side-by-side:
- **CF (Item-Item Cosine)** — similar = co-watched by the same users
- **SVD (Latent Factors)** — similar = close in 128-dim taste space
- **Two-Tower (Content + ID)** — similar = close in 128-dim text+genre+ID
  embedding (works for new movies)

Plus the source movie's interaction distribution by gender + age, so you can
see which audience signal each model is learning from.

### 3. Model Bias — Quantitative audit dashboard
Goal: surface popularity bias, demographic gaps, and genre concentration in
served recommendations *before* they reach users. Five audits per model
(popularity ratio, long-tail fraction, gender gap, age gap, genre
concentration), with explicit good/bad thresholds and an opinionated verdict
on each metric. Compares **SASRec** (raw) vs **Reranked** (post-LightGBM)
to show whether the reranker amplifies or mitigates each bias.

### 4. Cold Start — New item / new user flows
Two scenarios:
- **New item** — paste a title + genres, encode through the Two-Tower content
  pipeline, find similar existing items + the demographic profile of users
  whose vectors are closest. Pre-launch fairness check: would this item
  systematically be served to one demographic over another?
- **New user** — supply demographics only (zero history). Returns trending +
  Two-Tower demographic-only recommendations.

### 5. Training Runs — Weights & Biases
Embedded W&B report. Loss curves, eval metrics, and config for every training
run. Same data, different lens — for inspecting the experiment trail behind
the headline numbers.

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │     Streamlit UI  (port 7860)    │
                    └────────────────┬────────────────┘
                                     │ HTTP
                    ┌────────────────▼────────────────┐
                    │     FastAPI         (port 8000)  │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │ TieredRecommendationEngine       │
                    │                                  │
                    │  history_len = 0   → Tier 1     │
                    │     popularity → reranker        │
                    │                                  │
                    │  history_len 1–5  → Tier 2      │
                    │     CF + SVD candidates →        │
                    │     LightGBM reranker            │
                    │                                  │
                    │  history_len > 5   → Tier 3     │
                    │     SASRec direct (no rerank)    │
                    │                                  │
                    │  cold item / cold user           │
                    │     → Two-Tower content tower    │
                    │     → FAISS IndexFlatIP          │
                    └──────────────────────────────────┘
```

---

## Repository map

```
api/                FastAPI service (8 endpoints)
inference/          TieredRecommendationEngine, reranker, latency benchmark
models/             5 model implementations + trainers
  cf/               Item-item & user-user CF (implicit)
  svd/              ALS matrix factorization (implicit)
  two_tower/        Content + ID dual-encoder (PyTorch)
  sasrec/           Self-attentive sequential model (PyTorch)
  bert4rec/         Bidirectional Cloze model (PyTorch)
evaluation/         Metrics + bias audit
ui/                 Streamlit app (5 tabs)
data/               Download, preprocess, simulate, feature store
config/             config.yaml + loader
artifacts/          Trained models, FAISS index, embeddings, bias CSVs
scripts/verify/     Section-by-section verification scripts (build trail)
tests/              pytest unit tests
```

---

## Documentation

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** — full model details, hyperparameters, evaluation, bias findings
- **[SETUP.md](SETUP.md)** — recreate the analysis end-to-end on a fresh machine
- **[DEPLOY.md](DEPLOY.md)** — push to Hugging Face Spaces (Docker)

---

## Quick start (local)

```bash
git clone https://github.com/deepanvishal/RecSys
cd RecSys
pip install -r requirements.txt

# in one terminal
python -m api.main

# in another
streamlit run ui/app.py
```

API at <http://localhost:8000/docs>, UI at <http://localhost:8501>.

For the full reproduction (download + train + eval), see **[SETUP.md](SETUP.md)**.

---

## Stack

Python 3.10 · PyTorch 2.3 · FastAPI · Streamlit · LightGBM · FAISS-CPU ·
implicit · sentence-transformers · Weights & Biases · Docker

## License

MIT — see [LICENSE](LICENSE).
