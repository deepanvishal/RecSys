import numpy as np
import scipy.sparse as sp
from implicit.nearest_neighbours import CosineRecommender
from config.config_loader import get_config
from utils.logger import get_logger


logger = get_logger('item_item_cf')


class ItemItemCF:
    def __init__(self, K=20):
        self.K = K
        self.model = CosineRecommender(K=K)
        self.matrix = None

    def fit(self, matrix: sp.csr_matrix):
        # implicit >=0.5 expects user-items in fit() and recommend()
        self.matrix = matrix.tocsr()
        self.model.fit(self.matrix)
        logger.info(f'ItemItemCF fitted: K={self.K}, matrix={matrix.shape}, sim_nnz={self.model.similarity.nnz}')

    def recommend(self, user_id: int, N: int = 10) -> list:
        ids, _ = self.model.recommend(
            user_id, self.matrix[user_id], N=N,
            filter_already_liked_items=True,
        )
        return ids.tolist()

    def save(self, path: str):
        import joblib
        joblib.dump(self, path)
        logger.info(f'Saved to {path}')

    @staticmethod
    def load(path: str):
        import joblib
        return joblib.load(path)

    def retrieve(self, history: list, k: int = 100, exclude_ids: list = None) -> tuple:
        """
        Retrieve top-k candidate items.
        Returns: (item_ids, scores) — both np.ndarray of shape (<=k,).
        Excludes history (always) plus any extra exclude_ids.
        """
        import scipy.sparse as sp
        if not history:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        exclude = set(exclude_ids or []) | set(history)
        n_items = self.matrix.shape[1]
        col = np.asarray(history, dtype=np.int32)
        col = col[col < n_items]
        row = np.zeros(len(col), dtype=np.int32)
        data = np.ones(len(col), dtype=np.float32)
        user_mat = sp.csr_matrix((data, (row, col)), shape=(1, n_items))
        # implicit >=0.5 returns (ids_array, scores_array). Fetch extra to allow masking.
        ids, scores = self.model.recommend(
            userid=0, user_items=user_mat,
            N=k + len(exclude),
            filter_already_liked_items=False,
        )
        ids = np.asarray(ids, dtype=np.int32)
        scores = np.asarray(scores, dtype=np.float32)
        mask = ~np.isin(ids, list(exclude))
        return ids[mask][:k], scores[mask][:k]
