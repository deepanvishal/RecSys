import numpy as np
import pandas as pd
import pytest
from evaluation.metrics import hit_rate, ndcg, evaluate_model, evaluate_vectorized


# ---- hit_rate ----
def test_hit_rate_found():
    assert hit_rate([1, 2, 3, 4, 5], 3) == 1


def test_hit_rate_not_found():
    assert hit_rate([1, 2, 3, 4, 5], 9) == 0


def test_hit_rate_empty():
    assert hit_rate([], 1) == 0


# ---- ndcg ----
def test_ndcg_rank1():
    assert ndcg([5, 1, 2], 5) == pytest.approx(1.0)


def test_ndcg_rank2():
    expected = 1.0 / np.log2(3)
    assert ndcg([1, 5, 2], 5) == pytest.approx(expected)


def test_ndcg_not_found():
    assert ndcg([1, 2, 3], 9) == 0.0


# ---- evaluate_model ----
def test_evaluate_model_perfect():
    def perfect_recommend(uid, N):
        return [uid]
    val_df = pd.DataFrame({'user_id': [0, 1, 2], 'item_id': [0, 1, 2]})
    metrics = evaluate_model(perfect_recommend, val_df, K_list=[1, 5, 10])
    assert metrics['HR@1'] == 1.0
    assert metrics['HR@5'] == 1.0
    assert metrics['NDCG@1'] == 1.0


def test_evaluate_model_zero():
    def bad_recommend(uid, N):
        return [999, 998, 997]
    val_df = pd.DataFrame({'user_id': [0, 1, 2], 'item_id': [0, 1, 2]})
    metrics = evaluate_model(bad_recommend, val_df)
    assert metrics['HR@10'] == 0.0
    assert metrics['NDCG@10'] == 0.0


# ---- evaluate_vectorized ----
def test_evaluate_vectorized_perfect():
    np.random.seed(42)
    n_users, n_items = 10, 20
    scores = np.random.randn(n_users, n_items).astype(np.float32)
    gt_items = list(range(n_users))
    for u, i in enumerate(gt_items):
        scores[u, i] = 999.0
    val_df = pd.DataFrame({'user_id': range(n_users), 'item_id': gt_items})
    train_df = pd.DataFrame({'user_id': [], 'item_id': []})
    metrics = evaluate_vectorized(scores, val_df, train_df)
    assert metrics['HR@10'] == 1.0
    assert metrics['NDCG@10'] == pytest.approx(1.0)


def test_evaluate_vectorized_seen_mask():
    """Without mask, item 0 (highest score) would beat the ground truth (item 5).
    With mask, item 0 is removed from contention and ground truth ranks first."""
    n_users, n_items = 5, 10
    scores = np.zeros((n_users, n_items), dtype=np.float32)
    scores[:, 0] = 100.0   # would be top-1 — but it's a "seen" item
    scores[:, 5] = 50.0    # ground truth, second-best raw score

    val_df = pd.DataFrame({'user_id': range(n_users), 'item_id': [5] * n_users})

    # No mask — ground truth ranks 2nd
    no_mask = pd.DataFrame({'user_id': [], 'item_id': []})
    m_unmasked = evaluate_vectorized(scores.copy(), val_df, no_mask, K_list=[1, 5])
    assert m_unmasked['HR@1'] == 0.0   # item 0 wins, gt at rank 2
    assert m_unmasked['HR@5'] == 1.0   # gt is in top-5

    # With mask — item 0 removed, ground truth wins
    train_df = pd.DataFrame({'user_id': range(n_users), 'item_id': [0] * n_users})
    m_masked = evaluate_vectorized(scores.copy(), val_df, train_df, K_list=[1, 5])
    assert m_masked['HR@1'] == 1.0     # mask let gt rank first
    assert m_masked['NDCG@1'] == pytest.approx(1.0)
