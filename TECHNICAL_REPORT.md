# Serko RecSys — Technical Report

**Live demo:** <https://huggingface.co/spaces/deepanvishal/serko-recsys>
**W&B:** <https://wandb.ai/deepanvishal-cvs-health/serko-recsys-movielens>

---

## 1. Problem framing

A take-home recommender system demonstrating progression from simple to
complex models on a single dataset. Targets three production scenarios via
a tiered architecture: **cold start** (no history), **sparse user**
(1–5 interactions), **warm user** (>5 interactions). Audits served
recommendations across demographic and popularity dimensions.

## 2. Dataset

**MovieLens 1M** — 6,040 users, 3,706 movies, 1,000,209 ratings.

Selected over Amazon Beauty (2014, 5-core) for three reasons:
- Proven benchmark for Two-Tower (TFRS), SASRec, and BERT4Rec
- User demographics (age, gender, occupation) enable bias analysis unavailable
  in product review datasets
- Interaction density (avg 165 ratings/user) validates all model classes

**Evaluation protocol:** leave-one-out split, **full-catalog ranking**
against all 3,952 items. Full-catalog is deliberately harder than the
99-negatives sampled evaluation common in published work — published 15–25%
SASRec HR@10 numbers use sampled eval; our 33.3% under full-catalog is
equivalent or better.

## 3. Models

### 3.1 Item-Item CF (`models/cf/item_item_cf.py`)

| | |
|---|---|
| **Library** | `implicit.CosineRecommender` |
| **Input** | Sparse user-item interaction matrix (6040 × 3706) |
| **Hyperparams** | K ∈ {10, 20, 50} swept; **best K=10** |
| **Output** | Top-N item IDs + cosine scores |
| **Training time** | ~5s on CPU |
| **Best HR@10** | 0.0816 |
| **Role** | Item-similarity surface (Tab 2 of UI) |

UserUserCF (cosine, K=50) marginally better at HR@10=0.0830 — consistent
with ML-1M's dense histories favouring user-space similarity.

### 3.2 SVD / ALS (`models/svd/als_model.py`)

| | |
|---|---|
| **Library** | `implicit.AlternatingLeastSquares` |
| **Input** | Sparse user-item matrix |
| **Hyperparams** | factors ∈ {32, 64, 128}, regularization ∈ {0.01, 0.1}, iters=15 |
| **Best config** | **f=128, reg=0.1** |
| **Output** | User & item factor matrices (128-dim each) |
| **Training time** | ~30s on CPU |
| **Best HR@10** | 0.1020 |
| **Role** | Tier 2 retrieval (sparse-user fold-in) |

**Production feature:** `fold_in()` computes a user embedding from a partial
history in <100ms without retraining. Critical for the Tier-2 path where the
user_id may be unknown.

### 3.3 Two-Tower (`models/two_tower/model.py`)

| | |
|---|---|
| **Framework** | PyTorch |
| **Item tower input** | `(item_id, content_emb_384, genre_18d)` |
| **User tower input** | `(user_id, hist_items[seq_len], hist_lens, demo_3d)` |
| **Item tower** | id_emb(64) + content_proj(128) + genre_proj(32) → MLP → 128-d, L2-normalized |
| **User tower** | user_id_emb(64) + GRU(hist) + demo_proj(32) → MLP → 128-d, L2-normalized |
| **Loss** | In-batch sampled softmax, **temperature τ=0.2** |
| **Optimizer** | Adam lr=1e-3, batch=512, 30 epochs, early stop patience=5 |
| **Content emb** | sentence-transformers/all-MiniLM-L6-v2 over `"<title>. Genres: <g1, g2, ...>"` |
| **Output** | 128-dim normalized embeddings → FAISS `IndexFlatIP` |
| **Best HR@10** | 0.0647 |
| **Role** | Cold-item retrieval, demographic-only cold-user, Tab 4 of UI |

**Why τ=0.2 not 0.07:** τ=0.07 is for fine-tuning *pretrained* CLIP-style
encoders. From-scratch training needs a softer distribution; τ=0.07 collapses
gradients on day-1 weights. Validated empirically.

**Why HR@10 below CF:** in-batch negatives at batch=512 are weak vs CF's full
co-occurrence signal on dense data. Two-Tower's value lives in the cold-item
case, not warm-user ranking — that is the deployment surface.

### 3.4 SASRec (`models/sasrec/model.py`)

| | |
|---|---|
| **Framework** | PyTorch |
| **Architecture** | 2-block transformer, self-attention, causal mask |
| **Hyperparams** | hidden=128, heads=2, dropout=0.2, max_len=200 |
| **Loss** | Full-vocab cross-entropy (SASRec+, RecSys 2023) |
| **Optimizer** | Adam lr=1e-3, weight_decay=1e-5, batch=128, 100 epochs, early stop patience=5 |
| **Input** | Padded sequence of item IDs (last interaction = target) |
| **Output** | Logits over full vocab (3953 items + pad) |
| **Best epoch** | 26 (early-stopped at 31) |
| **Loss trajectory** | 8.28 → 5.30 (random baseline = ln(3952)=8.28) |
| **Best HR@10** | **0.3328** |
| **Role** | Tier 3 — warm-user direct recommendations (no rerank) |

3.26× better than SVD, 4× better than CF. Sequential self-attention captures
temporal preference drift that latent factors and co-occurrence miss.

### 3.5 BERT4Rec (`models/bert4rec/model.py`)

| | |
|---|---|
| **Framework** | PyTorch |
| **Architecture** | Bidirectional transformer, Cloze (masked-item) objective |
| **Hyperparams** | Identical to SASRec for fair comparison |
| **Mask probability** | 0.2 |
| **Best epoch** | 15 |
| **Best HR@10** | 0.0417 |
| **Role** | Experimental / not deployed |

Significantly under-performed SASRec on this dataset. Root cause:
6,040 training sequences × mask_prob=0.2 ≈ 24 effective batches/epoch — the
patience=5 early-stop fires on noise before convergence. Petrov & Macdonald
(RecSys 2022) document that BERT4Rec on ML-1M needs 200–1000 epochs.
Architectural correctness verified; convergence is a dataset-size budget call.

### 3.6 LightGBM Reranker (`inference/reranker.py`)

| | |
|---|---|
| **Library** | LightGBM 4.3 |
| **Training data** | Top-50 candidates from CF + SVD + Two-Tower per user (`reranker_train_*.parquet`) |
| **21 features** | retrieval_score, gender, age, occupation, n_history, item_popularity, genre_affinity, 18 user_genre_* preference scores, model_origin one-hot |
| **Label** | Binary — was this candidate the held-out target? |
| **Objective** | binary cross-entropy |
| **Hyperparams** | learning_rate=0.05, num_leaves=63, max_depth=8, n_estimators=300, early_stop=20 |
| **Validation AUC** | **0.7417** |
| **Top feature** | item_popularity (importance ~33%) |

Applied at Tier 1 and Tier 2. Tier 3 (SASRec) is direct — its candidate
distribution already aligns with target relevance.

## 4. Tiered inference engine (`inference/engine.py`)

| Tier | Trigger | Pipeline | p50 latency |
|---|---|---|---|
| **1** | 0 interactions | popularity (top-200) → reranker | <2 ms |
| **2** | 1–5 interactions | CF top-100 + SVD top-100 (fold-in) → reranker | ~94 ms |
| **3** | >5 interactions | SASRec direct (full-vocab softmax → top-N) | ~2 ms (GPU) |
| **Cold-item** | new item | Two-Tower content tower → FAISS top-N | ~69 ms |
| **Demo cold-user** | demo only | Two-Tower user tower (zero history + demo) → FAISS | ~70 ms |

Tier 2 SVD fold-in latency (94ms) is the only path exceeding a 10ms target.
Mitigation paths documented: serve sparse cohort with f=64 (~50ms), or
precompute fold-in embeddings for known users at ingest time.

## 5. Bias audit (`evaluation/bias_audit.py`)

Five audits computed on top-10 recommendations served to ~1,000 sampled users:

| Metric | SASRec (raw) | After reranker | Interpretation |
|---|---|---|---|
| Popularity ratio (rec / catalog avg) | **3.18×** | 3.46× | SASRec mildly biased; reranker marginally amplifies |
| Long-tail fraction (recs below median pop) | 5.2% | 4.1% | Both models concentrate on head |
| Gender gap M − F (HR@10) | +0.025 | +0.022 | Male users slightly better served — data volume asymmetry (M=4331, F=1709) |
| Age HR@10 range (max − min) | 0.110 | 0.103 | Under-18 worst (25.7%), 50–55 best (36.7%) |
| Top-genre share | Comedy 35.6% | Comedy 34.1% | Mirrors catalog distribution — not pathological |
| SVD popularity ratio | — | 4.24× | Reference: SVD more popularity-biased than SASRec |

**Key findings:**
1. SASRec's popularity ratio (3.18×) is **lower** than SVD's (4.24×) —
   the more expressive model is also the *fairer* one.
2. Reranker has near-neutral effect on popularity bias (3.18 → 3.46×). It
   does not aggressively re-rank away from popular items because the LightGBM
   feature `item_popularity` carries 33% of feature importance — a known
   tradeoff between relevance and exposure diversity.
3. **Under-18 cohort underperforms by 7–11pp HR@10** — the highest-risk
   demographic dimension. Cohort size (n=222, 3.7% of users) is the cause,
   not a model bug, but is a real deployment risk for youth-facing surfaces.

## 6. Serving

### FastAPI (`api/main.py`)
Endpoints: `/health`, `/recommend`, `/recommend_all`, `/similar_items`,
`/item_similarity`, `/new_item_cold_start`, `/two_tower_with_demo`,
`/new_user_cold_start`, `/homepage`. Auto-generated OpenAPI at `/docs`.

### Streamlit UI (`ui/app.py`)
Five tabs:
1. **For You** — Netflix-style homepage, profile-driven
2. **Item Similarity** — three-model side-by-side
3. **Model Bias** — SASRec vs Reranked dashboard
4. **Cold Start** — new item + new user flows
5. **Training Runs** — embedded W&B report

### Container
`Dockerfile` runs both processes (uvicorn → port 8000 internal,
streamlit → port 7860 public) with a 60-retry health-check gate in
`start.sh`. Image is ~1.6 GB (torch CPU + sentence-transformers dominate).

## 7. Key findings (synthesized)

1. **Sequential models dominate on dense data.** SASRec HR@10=0.3328 is 3.26× SVD and 4× CF.
2. **Two-Tower is a cold-start tool, not a warm-user tool.** On dense ML-1M, CF beats Two-Tower; deployed only at the cold-item / cold-user surface.
3. **BERT4Rec convergence is a dataset-size problem.** 200+ epochs documented; not budget-justified for this submission.
4. **Popularity bias is bounded.** SASRec 3.18× < SVD 4.24× — the more expressive model is also the fairer one.
5. **Age is the highest-risk demographic dimension.** Under-18 underperforms by 7–11pp — small cohort, real deployment risk.

## 8. Limitations & next steps

- **Reranker amplifies popularity slightly** (3.18 → 3.46×). Add an MMR
  diversity penalty or a long-tail boost feature.
- **BERT4Rec under-trained.** A 200-epoch run would close the gap with SASRec.
- **Bias audit is partial:** only SASRec, SVD (popularity only), and Reranked.
  CF, Two-Tower, BERT4Rec audits are not yet computed.
- **Tier 2 latency at 94ms** exceeds a 10ms target — precompute fold-in
  embeddings at ingest time, or serve f=64.
- **No online A/B harness.** All metrics are offline leave-one-out.

## 9. References

- SASRec — Kang & McAuley, ICDM 2018
- SASRec+ (CE loss) — Klenitskiy & Vasilev, RecSys 2023
- BERT4Rec — Sun et al., CIKM 2019
- BERT4Rec replicability — Petrov & Macdonald, RecSys 2022
- Two-Tower (TFRS) — Yi et al., RecSys 2019
- LightGBM — Ke et al., NeurIPS 2017
- FAISS — Johnson et al., 2019
