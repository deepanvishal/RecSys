import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.logger import get_logger
from utils.device import get_device
from models.svd.als_model import ALSModel
from models.sasrec.model import SASRec
from models.two_tower.model import TwoTowerModel
from models.two_tower.train_two_tower import build_full_item_tensors


logger = get_logger('inference_engine')


class TieredRecommendationEngine:
    """
    Tier 1 (cold,   0 interactions):  Popularity
    Tier 2 (sparse, 1-5 interactions): SVD fold-in
    Tier 3 (warm,   >5 interactions):  SASRec
    Cold-item fallback:                Two-Tower content retrieval
    """

    TIER2_THRESHOLD = 1
    TIER3_THRESHOLD = 5

    def __init__(self, cfg, processed_dir='data/processed', artifacts_dir='artifacts'):
        self.cfg = cfg
        self.device = get_device()
        processed = Path(processed_dir)
        artifacts = Path(artifacts_dir)

        stats = json.load(open(processed / 'dataset_stats.json'))
        self.n_users = stats['n_users']
        self.n_items = stats['n_items']

        logger.info('Loading inference engine...')
        self._load_popularity(processed)
        self._load_svd(artifacts)
        self._load_sasrec(artifacts)
        self._load_two_tower(artifacts, processed)
        logger.info('Engine ready.')

    def _load_popularity(self, processed):
        train = pd.read_parquet(processed / 'train.parquet')
        counts = train.groupby('item_id').size().sort_values(ascending=False)
        self.popular_items = counts.index.tolist()
        logger.info(f'Popularity: {len(self.popular_items)} items indexed')

    def _load_svd(self, artifacts):
        self.svd = ALSModel.load(str(artifacts / 'models/svd/als_best.pkl'))
        logger.info(f'SVD loaded: factors={self.svd.factors}')

    def _load_sasrec(self, artifacts):
        sr_cfg = self.cfg['models']['sasrec']
        self.sasrec = SASRec(
            n_items=self.n_items,
            hidden=sr_cfg['hidden'],
            max_len=sr_cfg['max_len'],
            num_blocks=sr_cfg['num_blocks'],
            num_heads=sr_cfg['num_heads'],
            dropout=0.0,
        ).to(self.device)
        self.sasrec.load_state_dict(
            torch.load(str(artifacts / 'models/sasrec/sasrec_best.pt'),
                       map_location=self.device),
        )
        self.sasrec.eval()
        self.sasrec_max_len = sr_cfg['max_len']
        logger.info('SASRec loaded.')

    def _load_two_tower(self, artifacts, processed):
        self.two_tower = TwoTowerModel(
            self.n_users, self.n_items,
            temperature=self.cfg['models']['two_tower']['temperature'],
        ).to(self.device)
        self.two_tower.load_state_dict(
            torch.load(str(artifacts / 'models/two_tower/two_tower_best.pt'),
                       map_location=self.device),
        )
        self.two_tower.eval()
        content_emb = np.load(str(artifacts / 'two_tower/item_content_emb.npy'))
        genre_matrix = np.load(str(artifacts / 'two_tower/item_genre_matrix.npy'))
        self.tt_item_ids, self.tt_content, self.tt_genres = build_full_item_tensors(
            content_emb, genre_matrix, self.device,
        )
        logger.info('Two-Tower loaded.')

    def recommend(self, user_id, history, N: int = 10, exclude_seen: bool = True) -> dict:
        """
        Main entry point.
        user_id: int (informational only — engine routes by history length, not id)
        history: list of item_ids the user interacted with (chronological)
        N: number of recommendations
        Returns: items, tier (1/2/3), model name, latency_ms, n_history.
        """
        t0 = time.perf_counter()
        n_history = len(history)

        if n_history == 0:
            items, tier, model = self._tier1(N)
        elif n_history <= self.TIER3_THRESHOLD:
            items, tier, model = self._tier2(history, N, exclude_seen)
        else:
            items, tier, model = self._tier3(history, N, exclude_seen)

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            'items': items,
            'tier': tier,
            'model': model,
            'latency_ms': round(latency_ms, 2),
            'n_history': n_history,
        }

    def _tier1(self, N):
        return self.popular_items[:N], 1, 'popularity'

    def _tier2(self, history, N, exclude_seen):
        emb = self.svd.fold_in(history)
        seen = history if exclude_seen else []
        items = self.svd.recommend_from_embedding(emb, seen, N=N)
        return items, 2, 'svd_foldin'

    def _tier3(self, history, N, exclude_seen):
        max_len = self.sasrec_max_len
        seq = [i + 1 for i in history[-max_len:]]   # +1 shift for SASRec vocab (0=pad)
        padded = [0] * (max_len - len(seq)) + seq
        seq_t = torch.tensor([padded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.sasrec.score_all_items(seq_t).squeeze(0).cpu().numpy()
        if exclude_seen:
            logits[history] = -np.inf
        items = np.argsort(logits)[::-1][:N].tolist()
        return items, 3, 'sasrec'

    def recommend_cold_item(self, item_id: int, N: int = 10) -> dict:
        """Retrieve similar items via Two-Tower item embeddings (handles unseen items)."""
        t0 = time.perf_counter()
        item_t = torch.tensor([item_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            query_emb = self.two_tower.encode_items(
                item_t, self.tt_content[item_t], self.tt_genres[item_t],
            )
            all_embs = self.two_tower.encode_items(
                self.tt_item_ids, self.tt_content, self.tt_genres,
            )
        scores = (query_emb @ all_embs.T).squeeze(0).cpu().numpy()
        scores[item_id] = -np.inf
        items = np.argsort(scores)[::-1][:N].tolist()
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            'items': items, 'tier': 'cold_item',
            'model': 'two_tower', 'latency_ms': round(latency_ms, 2),
        }
