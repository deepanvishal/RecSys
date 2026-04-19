import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import get_logger


logger = get_logger('user_user_cf')


class UserUserCF:
    def __init__(self, K=50):
        self.K = K  # top-K similar users
        self.matrix = None

    def fit(self, matrix: sp.csr_matrix):
        self.matrix = matrix
        logger.info(f'UserUserCF ready: K={self.K}, matrix={matrix.shape}')

    def recommend(self, user_id: int, N: int = 10) -> list:
        user_vec = self.matrix[user_id]
        sims = cosine_similarity(user_vec, self.matrix).flatten()
        sims[user_id] = -1  # exclude self
        top_users = np.argsort(sims)[::-1][:self.K]
        scores = np.asarray(self.matrix[top_users].sum(axis=0)).flatten()
        seen = user_vec.nonzero()[1]
        scores[seen] = -1
        top_items = np.argsort(scores)[::-1][:N]
        return top_items.tolist()
