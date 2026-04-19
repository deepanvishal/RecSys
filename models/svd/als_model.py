import numpy as np
import scipy.sparse as sp
import joblib
from implicit.als import AlternatingLeastSquares
from utils.logger import get_logger


logger = get_logger('als_model')


class ALSModel:
    def __init__(self, factors=64, regularization=0.01, iterations=50):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False,
            num_threads=0,
            calculate_training_loss=True,
        )
        self.user_factors = None
        self.item_factors = None
        self.user_item_matrix = None
        self.item_user_matrix = None

    def fit(self, item_user_matrix: sp.csr_matrix):
        """
        Args:
            item_user_matrix: (n_items, n_users) sparse — file convention from feature_store.
        Note: implicit >=0.5 expects user_item in fit(), so we transpose at this boundary.
        """
        self.item_user_matrix = item_user_matrix
        self.user_item_matrix = item_user_matrix.T.tocsr()
        self.model.fit(self.user_item_matrix)
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        logger.info(f'ALS fitted: factors={self.factors}, reg={self.regularization}')
        logger.info(f'  user_factors: {self.user_factors.shape}')
        logger.info(f'  item_factors: {self.item_factors.shape}')

    def recommend(self, user_id: int, N: int = 10) -> list:
        ids, _ = self.model.recommend(
            user_id,
            self.user_item_matrix[user_id],
            N=N,
            filter_already_liked_items=True,
        )
        return ids.tolist()

    def fold_in(self, item_ids: list, ratings: list = None) -> np.ndarray:
        """
        Compute user embedding from partial interaction history.
        Used for Tier 2 sparse users at inference time.
        """
        if ratings is None:
            ratings = [1.0] * len(item_ids)
        n_items = self.item_factors.shape[0]
        sparse_user = sp.csr_matrix(
            ([float(r) for r in ratings], ([0] * len(item_ids), item_ids)),
            shape=(1, n_items),
        )
        user_embedding = self.model.recalculate_user(0, sparse_user)
        return np.asarray(user_embedding).flatten()

    def recommend_from_embedding(self, user_embedding: np.ndarray,
                                  seen_items: list, N: int = 10) -> list:
        """Rank all items by dot product with user embedding, exclude seen."""
        scores = np.asarray(self.item_factors @ user_embedding).flatten()
        scores[seen_items] = -np.inf
        return np.argsort(scores)[::-1][:N].tolist()

    def score_all_users(self) -> np.ndarray:
        """
        (n_users, n_items) score matrix via single matmul.
        For ML-1M: (6040, 128) @ (3952, 128).T = (6040, 3952), ~95MB float32, <1s.
        """
        return (self.user_factors @ self.item_factors.T).astype(np.float32)

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f'Saved ALS model to {path}')

    @staticmethod
    def load(path: str):
        return joblib.load(path)

    def retrieve(self, history: list, k: int = 100, exclude_ids: list = None) -> tuple:
        """
        Retrieve top-k candidates via fold-in. Works for any history length.
        Returns: (item_ids, scores) — both np.ndarray of shape (<=k,).
        """
        if not history:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        exclude = set(exclude_ids or []) | set(history)
        user_vec = self.fold_in(history)                                         # (factors,)
        scores_arr = np.asarray(self.item_factors @ user_vec).flatten().astype(np.float32)
        if exclude:
            scores_arr[list(exclude)] = -np.inf
        sorted_ids = np.argsort(scores_arr)[::-1][:k]
        return sorted_ids.astype(np.int32), scores_arr[sorted_ids]
