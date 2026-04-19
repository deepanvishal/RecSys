import numpy as np
import faiss
from pathlib import Path
from config.config_loader import get_config
from utils.logger import get_logger
from utils.device import get_device


logger = get_logger('retrieval')


class FAISSRetriever:
    """
    FAISS IndexFlatIP on normalized item embeddings.
    FlatIP = exact inner product = cosine similarity (embeddings are L2-normalized).
    GPU-accelerated on RTX 3090 when available.
    """

    def __init__(self):
        self.cfg = get_config()
        self.index = None
        self.index_path = Path(self.cfg['faiss']['index_path'])
        self.top_k = self.cfg['faiss']['top_k_retrieval']
        self.dim = self.cfg['models']['two_tower']['item_embedding_dim']
        self.n_items = None

    def build(self, item_embeddings: np.ndarray):
        self.n_items = len(item_embeddings)
        logger.info(f'Building FAISS FlatIP index: {item_embeddings.shape}')
        device = get_device()
        cpu_index = faiss.IndexFlatIP(self.dim)
        if device.type == 'cuda':
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self._gpu_res = res
                logger.info('FAISS index on GPU')
            except Exception as e:
                logger.warning(f'Falling back to CPU FAISS index ({e})')
                self.index = cpu_index
        else:
            self.index = cpu_index
            logger.info('FAISS index on CPU')
        faiss.normalize_L2(item_embeddings)
        self.index.add(item_embeddings)
        logger.info(f'FAISS index built: {self.index.ntotal} items')

    def search(self, query_vectors: np.ndarray, k: int = None):
        assert self.index is not None, 'Build index first.'
        if k is None:
            k = self.top_k
        faiss.normalize_L2(query_vectors)
        scores, indices = self.index.search(query_vectors, k)
        return scores, indices

    def save(self):
        self.index_path.mkdir(parents=True, exist_ok=True)
        save_path = self.index_path / 'two_tower.index'
        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index
        faiss.write_index(cpu_index, str(save_path))
        logger.info(f'FAISS index saved to {save_path}')

    def load(self):
        load_path = self.index_path / 'two_tower.index'
        logger.info(f'Loading FAISS index from {load_path}')
        cpu_index = faiss.read_index(str(load_path))
        device = get_device()
        if device.type == 'cuda':
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self._gpu_res = res
            except Exception as e:
                logger.warning(f'Loading on CPU ({e})')
                self.index = cpu_index
        else:
            self.index = cpu_index
        self.n_items = self.index.ntotal
        logger.info(f'FAISS index loaded: {self.n_items} items')
        return self
