import numpy as np
from typing import List, Dict, Any


def hit_at_k(ground_truth_item: int, recommended_items: List[int], k: int) -> float:
    return 1.0 if ground_truth_item in recommended_items[:k] else 0.0


def ndcg_at_k(ground_truth_item: int, recommended_items: List[int], k: int) -> float:
    if ground_truth_item not in recommended_items[:k]:
        return 0.0
    rank = recommended_items[:k].index(ground_truth_item) + 1
    return 1.0 / np.log2(rank + 1)


def mrr(ground_truth_item: int, recommended_items: List[int]) -> float:
    if ground_truth_item not in recommended_items:
        return 0.0
    rank = recommended_items.index(ground_truth_item) + 1
    return 1.0 / rank


def catalog_coverage(recommended_items: List[int], n_total_items: int) -> float:
    """Fraction of catalog recommended at least once across all users."""
    return len(set(recommended_items)) / n_total_items


def novelty(recommended_items: List[int], popularity_scores: np.ndarray) -> float:
    """Average inverse popularity — higher = more novel."""
    pops = [popularity_scores[i] for i in recommended_items if i < len(popularity_scores)]
    if not pops:
        return 0.0
    return float(np.mean([-np.log2(p + 1e-10) for p in pops]))


def intra_list_diversity(recommended_items: List[int], item_embeddings: np.ndarray) -> float:
    """Average pairwise cosine distance between recommended items."""
    if len(recommended_items) < 2:
        return 0.0
    embs = item_embeddings[recommended_items]
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs_norm = embs / norms
    sim_matrix = embs_norm @ embs_norm.T
    n = len(recommended_items)
    diversity = (1 - sim_matrix).sum() / (n * (n - 1))
    return float(diversity)


def popularity_bias_index(recommended_items: List[int], popularity_scores: np.ndarray) -> float:
    """Mean popularity of recommended items. Higher = more biased toward popular."""
    pops = [popularity_scores[i] for i in recommended_items if i < len(popularity_scores)]
    return float(np.mean(pops)) if pops else 0.0


def evaluate_all(
    ground_truth: Dict[int, int],
    recommendations: Dict[int, List[int]],
    k_values: List[int],
    n_total_items: int,
    popularity_scores: np.ndarray,
    item_embeddings: np.ndarray = None,
) -> Dict[str, float]:
    """Run full evaluation suite. Returns dict of all metrics."""
    metrics = {}

    for k in k_values:
        hits, ndcgs, mrrs = [], [], []
        for user_id, gt_item in ground_truth.items():
            recs = recommendations.get(user_id, [])
            hits.append(hit_at_k(gt_item, recs, k))
            ndcgs.append(ndcg_at_k(gt_item, recs, k))
            if k == max(k_values):
                mrrs.append(mrr(gt_item, recs))
        metrics[f'hit@{k}'] = float(np.mean(hits))
        metrics[f'ndcg@{k}'] = float(np.mean(ndcgs))
        if k == max(k_values):
            metrics['mrr'] = float(np.mean(mrrs))

    max_k = max(k_values)
    all_recs_flat = [item for recs in recommendations.values() for item in recs[:max_k]]
    metrics['coverage'] = catalog_coverage(all_recs_flat, n_total_items)
    metrics['popularity_bias'] = popularity_bias_index(all_recs_flat, popularity_scores)
    metrics['novelty'] = novelty(all_recs_flat, popularity_scores)
    if item_embeddings is not None:
        sample_recs = all_recs_flat[:1000]
        metrics['diversity'] = intra_list_diversity(sample_recs, item_embeddings)
    return metrics
