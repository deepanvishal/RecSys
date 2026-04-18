import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from pathlib import Path
import joblib
from implicit.nearest_neighbours import CosineRecommender
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('cf')


class CollaborativeFilter:
    """
    Top-K sparse Collaborative Filtering for implicit feedback.
    - 'item': item-item top-K cosine similarity via implicit.nearest_neighbours.CosineRecommender.
              Bounded memory: stores at most K neighbours per item.
    - 'user': caches L2-normalized user-item matrix; computes top-K similar users on-the-fly per query.
              Bounded memory: never materializes a user-user similarity matrix.
    Cold start (zero history) explicitly falls back to popularity ranking.
    """

    def __init__(self, model_type='item', k=20):
        assert model_type in ('user', 'item'), "model_type must be 'user' or 'item'"
        self.model_type = model_type
        self.k = k
        self.cfg = get_config()
        self.interaction_matrix = None       # (n_users, n_items) csr
        self._item_model = None              # implicit CosineRecommender (item variant)
        self._user_normalized = None         # (n_users, n_items) csr, L2-normalized rows (user variant)
        self._user_normalized_T = None       # csc transpose for fast left-multiply
        self.popularity_scores = None        # (n_items,) float32
        self.n_users = None
        self.n_items = None
        self.is_fitted = False

    def fit(self, interaction_matrix: sp.csr_matrix):
        self.interaction_matrix = interaction_matrix.tocsr().astype(np.float32)
        self.n_users, self.n_items = self.interaction_matrix.shape
        logger.info(f'Fitting {self.model_type}-{self.model_type} CF, K={self.k}')
        sparsity = 1 - self.interaction_matrix.nnz / (self.n_users * self.n_items)
        logger.info(f'Matrix shape: {self.interaction_matrix.shape}, nnz: {self.interaction_matrix.nnz}, sparsity: {sparsity:.6f}')

        item_counts = np.asarray(self.interaction_matrix.sum(axis=0)).flatten()
        self.popularity_scores = (item_counts / (item_counts.max() + 1e-8)).astype(np.float32)

        if self.model_type == 'item':
            logger.info('Building top-K item-item cosine similarity (implicit)...')
            self._item_model = CosineRecommender(K=self.k)
            self._item_model.fit(self.interaction_matrix, show_progress=True)
            logger.info(f'Item-item similarity nnz: {self._item_model.similarity.nnz}')
        else:
            logger.info('Caching L2-normalized user-item matrix for on-the-fly user-user CF...')
            self._user_normalized = normalize(self.interaction_matrix, norm='l2').tocsr()
            self._user_normalized_T = self._user_normalized.T.tocsc()

        self.is_fitted = True
        logger.info('Fitting complete.')
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True):
        """Top-N recommendations. Cold-start users get popularity fallback."""
        assert self.is_fitted, 'Model not fitted. Call fit() first.'
        user_row = self.interaction_matrix[user_id]
        if user_row.nnz == 0:
            return self._popularity_fallback(n, exclude_ids=[])

        if self.model_type == 'item':
            ids, scores = self._item_model.recommend(
                userid=user_id,
                user_items=user_row,
                N=n,
                filter_already_liked_items=exclude_seen,
            )
            return [(int(i), float(s)) for i, s in zip(ids, scores)]
        return self._user_based_recommend(user_id, user_row, n, exclude_seen)

    def _user_based_recommend(self, user_id, user_row, n, exclude_seen):
        # Per-query similarity to all users — no N×N matrix ever materialized.
        user_vec = normalize(user_row, norm='l2')                      # (1, n_items) sparse
        sims = np.asarray((user_vec @ self._user_normalized_T).todense()).flatten()  # (n_users,)
        sims[user_id] = 0.0
        if self.k < len(sims):
            top_k_users = np.argpartition(sims, -self.k)[-self.k:]
        else:
            top_k_users = np.arange(len(sims))
        weights = sims[top_k_users]
        neighbor_mat = self.interaction_matrix[top_k_users]            # (K, n_items)
        scores = np.asarray(weights @ neighbor_mat).flatten()          # (n_items,)
        if exclude_seen:
            scores[user_row.indices] = -np.inf
        return self._top_n(scores, n)

    def _popularity_fallback(self, n: int, exclude_ids: list):
        scores = self.popularity_scores.copy()
        if exclude_ids:
            scores[exclude_ids] = -np.inf
        return self._top_n(scores, n)

    @staticmethod
    def _top_n(scores: np.ndarray, n: int):
        if n >= len(scores):
            top_idx = np.argsort(scores)[::-1][:n]
        else:
            top_idx = np.argpartition(scores, -n)[-n:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(i), float(scores[i])) for i in top_idx]

    def get_trending(self, n: int = 10):
        """Top-N trending items by popularity. Used by tiered engine Tier 1 (cold start)."""
        assert self.is_fitted
        return self._popularity_fallback(n, exclude_ids=[])

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path) / f'cf_{self.model_type}_k{self.k}.pkl')
        logger.info(f'Model saved to {path}')

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
