import numpy as np


def hit_rate(recommended: list, ground_truth: int) -> int:
    return int(ground_truth in recommended)


def ndcg(recommended: list, ground_truth: int) -> float:
    if ground_truth not in recommended:
        return 0.0
    rank = recommended.index(ground_truth) + 1
    return 1.0 / np.log2(rank + 1)


def evaluate_model(recommend_fn, val_df, K_list=[5, 10, 20]) -> dict:
    """
    Per-user loop. Use for models that don't expose a cheap full score matrix
    (e.g. UserUserCF on-the-fly cosine, Two-Tower, BERT4Rec).
    recommend_fn(user_id, N) -> list[item_id]
    val_df: one row per user, columns user_id + item_id (ground truth)
    """
    results = {f'HR@{k}': [] for k in K_list}
    results.update({f'NDCG@{k}': [] for k in K_list})

    for _, row in val_df.iterrows():
        uid = int(row['user_id'])
        gt = int(row['item_id'])
        max_k = max(K_list)
        recs = recommend_fn(uid, max_k)
        for k in K_list:
            results[f'HR@{k}'].append(hit_rate(recs[:k], gt))
            results[f'NDCG@{k}'].append(ndcg(recs[:k], gt))

    return {metric: float(np.mean(vals)) for metric, vals in results.items()}


def evaluate_vectorized(score_matrix: np.ndarray,
                         val_df,
                         train_df,
                         K_list: list = [5, 10, 20]) -> dict:
    """
    Vectorized leave-one-out evaluation.
    score_matrix: (n_users, n_items) float32 — raw scores, higher = better.
    val_df:   columns [user_id, item_id] — one row per user (ground truth).
    train_df: columns [user_id, item_id] — used to mask seen items.
    Rank = number of items scoring >= ground truth (ties get worst rank, conservative).
    """
    scores = score_matrix.copy()  # (n_users, n_items)
    if len(train_df) > 0:
        seen_pairs = train_df[['user_id', 'item_id']].values.astype(np.int64)
        scores[seen_pairs[:, 0], seen_pairs[:, 1]] = -np.inf

    val_users = val_df['user_id'].values.astype(int)
    val_items = val_df['item_id'].values.astype(int)

    user_scores = scores[val_users]                                          # (n_val, n_items)
    gt_scores = user_scores[np.arange(len(val_users)), val_items]            # (n_val,)
    ranks = (user_scores >= gt_scores[:, None]).sum(axis=1)                  # (n_val,)

    results = {}
    for k in K_list:
        hits = (ranks <= k).astype(float)
        ndcg_scores = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
        results[f'HR@{k}'] = float(hits.mean())
        results[f'NDCG@{k}'] = float(ndcg_scores.mean())

    return results
