import numpy as np
import scipy.sparse as sp
import joblib
from pathlib import Path
from implicit.als import AlternatingLeastSquares
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('svd')


class SVDModel:
    """
    SVD via Alternating Least Squares (implicit lib).
    Key capability: fold-in for sparse/new users without full retrain.
    Tier 2 of the tiered inference engine uses this model.
    """

    def __init__(self, factors: int = 128):
        self.cfg = get_config()
        svd_cfg = self.cfg['models']['svd']
        self.factors = factors
        self.regularization = svd_cfg['regularization']
        self.iterations = svd_cfg['iterations']
        self.num_threads = svd_cfg['num_threads']
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            num_threads=self.num_threads,
            use_gpu=False,
            calculate_training_loss=True,
        )
        self.item_user_matrix = None
        self.user_item_matrix = None
        self.popularity_scores = None
        self.n_users = None
        self.n_items = None
        self.is_fitted = False

    def fit(self, item_user_matrix: sp.csr_matrix):
        """
        Fit ALS model.
        Args:
            item_user_matrix: sparse (n_items, n_users) — file convention from feature_store
        Note: implicit >=0.5 expects user-item in fit(); we transpose at this boundary.
        """
        self.item_user_matrix = item_user_matrix
        self.user_item_matrix = item_user_matrix.T.tocsr()
        self.n_items = item_user_matrix.shape[0]
        self.n_users = item_user_matrix.shape[1]
        logger.info(f'Fitting ALS: factors={self.factors}, iterations={self.iterations}')
        logger.info(f'Input item_user shape: {item_user_matrix.shape}, nnz: {item_user_matrix.nnz}')
        logger.info(f'Passing user_item shape {self.user_item_matrix.shape} to implicit.fit()')
        self.model.fit(self.user_item_matrix)
        item_counts = np.asarray(item_user_matrix.sum(axis=1)).flatten()
        self.popularity_scores = item_counts / (item_counts.max() + 1e-8)
        self.is_fitted = True
        logger.info(f'ALS fit complete. user_factors={self.model.user_factors.shape}, item_factors={self.model.item_factors.shape}')
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True):
        """Recommend top-N items. Popularity fallback for zero-history users."""
        assert self.is_fitted, 'Call fit() first.'
        user_interactions = self.user_item_matrix[user_id]
        if user_interactions.nnz == 0:
            return self._popularity_fallback(n)
        ids, scores = self.model.recommend(
            user_id,
            self.user_item_matrix[user_id],
            N=n,
            filter_already_liked_items=exclude_seen,
        )
        return list(zip(ids.tolist(), scores.tolist()))

    def foldin_recommend(self, interaction_items: list, n: int = 10):
        """
        Fold-in: recommend for a sparse/new user without retraining.
        O(factors^2) per query — not O(n_users).
        """
        assert self.is_fitted, 'Call fit() first.'
        if not interaction_items:
            return self._popularity_fallback(n)
        data = np.ones(len(interaction_items), dtype=np.float32)
        row = np.zeros(len(interaction_items), dtype=np.int32)
        col = np.array(interaction_items, dtype=np.int32)
        valid = col < self.n_items
        data, row, col = data[valid], row[valid], col[valid]
        user_vector = sp.csr_matrix(
            (data, (row, col)),
            shape=(1, self.n_items),
        )
        user_embedding = self.model.recalculate_user(0, user_vector)
        user_embedding = np.asarray(user_embedding).flatten()
        scores = np.asarray(self.model.item_factors @ user_embedding).flatten()
        scores[col] = -np.inf  # mask seen
        top_n = np.argpartition(scores, -n)[-n:]
        top_n = top_n[np.argsort(scores[top_n])[::-1]]
        return [(int(i), float(scores[i])) for i in top_n]

    def get_similar_items(self, item_id: int, n: int = 10):
        assert self.is_fitted
        ids, scores = self.model.similar_items(item_id, N=n + 1)
        return [(int(i), float(s)) for i, s in zip(ids, scores) if i != item_id][:n]

    def get_trending(self, n: int = 10):
        assert self.is_fitted
        return self._popularity_fallback(n)

    def _popularity_fallback(self, n: int):
        top_n = np.argpartition(self.popularity_scores, -n)[-n:]
        top_n = top_n[np.argsort(self.popularity_scores[top_n])[::-1]]
        return [(int(i), float(self.popularity_scores[i])) for i in top_n]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        save_path = Path(path) / f'svd_factors{self.factors}.pkl'
        joblib.dump(self, save_path)
        logger.info(f'SVD model saved to {save_path}')

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
