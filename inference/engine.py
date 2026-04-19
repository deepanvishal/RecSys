import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.logger import get_logger
from utils.device import get_device
from models.cf.item_item_cf import ItemItemCF
from models.svd.als_model import ALSModel
from models.sasrec.model import SASRec
from models.bert4rec.model import BERT4Rec
from models.two_tower.model import TwoTowerModel
from models.two_tower.train_two_tower import build_full_item_tensors
from inference.reranker import Reranker
from inference.reranker_data import build_features, _build_lookups


logger = get_logger('inference_engine')


GENRES = ['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']

AGE_LABELS = {0: '<18', 1: '18-24', 2: '25-34', 3: '35-44',
              4: '45-49', 5: '50-55', 6: '56+'}


class TieredRecommendationEngine:
    """
    Tier 1 (0 interactions):   Popularity -> reranker
    Tier 2 (1-5 interactions): CF+SVD retrieve -> reranker
    Tier 3 (>5 interactions):  SASRec direct (no reranker)
    All-models:                5 models in parallel + reranked column
    Item similarity:           CF similarity matrix, SVD item_factors, Two-Tower item_emb
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
        self._load_cf(artifacts)
        self._load_svd(artifacts)
        self._load_sasrec(artifacts)
        self._load_bert4rec(artifacts)
        self._load_two_tower(artifacts, processed)
        self._load_reranker(artifacts)
        self._load_lookups(processed)
        logger.info('Engine ready.')

    def _load_popularity(self, processed):
        train = pd.read_parquet(processed / 'train.parquet')
        counts = train.groupby('item_id').size().sort_values(ascending=False)
        self.popular_items = counts.index.tolist()
        logger.info(f'Popularity: {len(self.popular_items)} items indexed')

    def _load_cf(self, artifacts):
        self.cf = ItemItemCF.load(str(artifacts / 'models/cf/item_item_best.pkl'))
        logger.info(f'CF loaded: K={self.cf.K}')

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

    def _load_bert4rec(self, artifacts):
        br_cfg = self.cfg['models']['bert4rec']
        self.bert4rec = BERT4Rec(
            n_items=self.n_items,
            hidden=br_cfg['hidden'],
            max_len=br_cfg['max_len'],
            num_blocks=br_cfg['num_blocks'],
            num_heads=br_cfg['num_heads'],
            dropout=0.0,
        ).to(self.device)
        self.bert4rec.load_state_dict(
            torch.load(str(artifacts / 'models/bert4rec/bert4rec_best.pt'),
                       map_location=self.device),
        )
        self.bert4rec.eval()
        self.bert4rec_max_len = br_cfg['max_len']
        self.bert4rec_mask = self.n_items + 1
        logger.info('BERT4Rec loaded.')

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
        self.tt_item_embs = np.load(str(artifacts / 'two_tower/item_emb.npy'))
        self.tt_user_embs = np.load(str(artifacts / 'two_tower/user_emb.npy'))
        logger.info('Two-Tower loaded.')

    def _load_reranker(self, artifacts):
        self.reranker = Reranker(str(artifacts / 'reranker/reranker.pkl'))
        logger.info('Reranker loaded.')

    def _load_lookups(self, processed):
        train_df = pd.read_parquet(processed / 'train.parquet')
        user_meta = pd.read_parquet(processed / 'user_metadata.parquet')
        item_meta = pd.read_parquet(processed / 'item_metadata.parquet')
        (
            self.user_history, self.n_history_map, self.item_genre,
            self.user_genre, self.user_demo, self.item_pop,
        ) = _build_lookups(train_df, user_meta, item_meta, self.n_items)
        logger.info('Lookups loaded.')

    # -------------------------------------------------------------------------
    # Public entry points
    # -------------------------------------------------------------------------

    def recommend(self, user_id, history, N: int = 10, exclude_seen: bool = True) -> dict:
        t0 = time.perf_counter()
        n_history = len(history)
        uid = user_id if user_id is not None else -1

        if n_history == 0:
            items, tier, model = self._tier1(uid, N)
        elif n_history <= self.TIER3_THRESHOLD:
            items, tier, model = self._tier2(uid, history, N, exclude_seen)
        else:
            items, tier, model = self._tier3(history, N, exclude_seen)

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            'items': items, 'tier': tier, 'model': model,
            'latency_ms': round(latency_ms, 2), 'n_history': n_history,
        }

    def recommend_all_models(self, user_id, history, N: int = 10) -> dict:
        """Run all 5 models for one user + reranked column."""
        n_history = len(history)
        uid = user_id if user_id is not None else -1
        tier = 1 if n_history == 0 else (2 if n_history <= self.TIER3_THRESHOLD else 3)

        if n_history == 0:
            pop = self.popular_items[:N]
            pop_sc = list(range(N, 0, -1))
            cands = self.popular_items[:100]
            cscores = np.arange(100, 0, -1, dtype=np.float32)
            feat = build_features(uid, cands, cscores,
                                  self.user_demo, self.n_history_map,
                                  self.user_genre, self.item_genre, self.item_pop)
            reranked = self.reranker.rerank(feat, k=N)
            return {
                'cf':        {'items': pop, 'scores': pop_sc},
                'svd':       {'items': pop, 'scores': pop_sc},
                'two_tower': {'items': pop, 'scores': pop_sc},
                'sasrec':    {'items': pop, 'scores': pop_sc},
                'bert4rec':  {'items': pop, 'scores': pop_sc},
                'reranked':  {'items': reranked, 'scores': list(range(N, 0, -1))},
                'tier': 1, 'n_history': 0,
            }

        cf_ids, cf_sc = self.cf.retrieve(history, k=100, exclude_ids=history)
        svd_ids, svd_sc = self.svd.retrieve(history, k=100, exclude_ids=history)
        tt_items = self._two_tower_for_user(user_id, history, N)
        sasrec_items = self._sasrec_score(history, N)
        bert_items = self._bert4rec_score(history, N)

        # Reranked: CF+SVD merged -> reranker (Tier 1/2 only).
        # Tier 3: reranked mirrors SASRec direct (no reranker on sequential models).
        if tier <= 2:
            merged = {}
            for iid, sc in zip(cf_ids, cf_sc):
                merged[int(iid)] = float(sc)
            for iid, sc in zip(svd_ids, svd_sc):
                k = int(iid)
                if k not in merged or float(sc) > merged[k]:
                    merged[k] = float(sc)
            cids = np.array(list(merged.keys()), dtype=np.int32)
            cscores = np.array(list(merged.values()), dtype=np.float32)
            feat = build_features(uid, cids, cscores,
                                  self.user_demo, self.n_history_map,
                                  self.user_genre, self.item_genre, self.item_pop)
            reranked = self.reranker.rerank(feat, k=N)
        else:
            reranked = sasrec_items

        return {
            'cf':        {'items': cf_ids[:N].tolist(),  'scores': cf_sc[:N].tolist()},
            'svd':       {'items': svd_ids[:N].tolist(), 'scores': svd_sc[:N].tolist()},
            'two_tower': {'items': tt_items,             'scores': list(range(N, 0, -1))},
            'sasrec':    {'items': sasrec_items,         'scores': list(range(N, 0, -1))},
            'bert4rec':  {'items': bert_items,           'scores': list(range(N, 0, -1))},
            'reranked':  {'items': reranked,             'scores': list(range(N, 0, -1))},
            'tier': tier, 'n_history': n_history,
        }

    def item_similarity(self, item_id: int, N: int = 10) -> dict:
        """Top-N similar items from CF, SVD, Two-Tower."""
        cf_row = np.asarray(self.cf.model.similarity[item_id].todense()).flatten()
        cf_row[item_id] = -np.inf
        cf_top = np.argsort(cf_row)[::-1][:N]
        cf_sc = cf_row[cf_top]

        svd_scores = np.asarray(self.svd.item_factors @ self.svd.item_factors[item_id]).flatten()
        svd_scores[item_id] = -np.inf
        svd_top = np.argsort(svd_scores)[::-1][:N]
        svd_sc = svd_scores[svd_top]

        tt_scores = self.tt_item_embs @ self.tt_item_embs[item_id]
        tt_scores[item_id] = -np.inf
        tt_top = np.argsort(tt_scores)[::-1][:N]
        tt_sc = tt_scores[tt_top]

        return {
            'cf':        {'items': cf_top.tolist(),  'scores': cf_sc.tolist()},
            'svd':       {'items': svd_top.tolist(), 'scores': svd_sc.tolist()},
            'two_tower': {'items': tt_top.tolist(),  'scores': tt_sc.tolist()},
        }

    def new_item_cold_start(self, title: str, genres: list,
                             n_similar: int = 10, n_users: int = 20) -> dict:
        """
        Encode a brand-new item through the Two-Tower item tower (no interaction
        history needed) and report similar existing items + likely users + audience bias.
        item_id=0 is the padding row -> id_emb contributes a zero vector;
        content (384) + genres (18) carry the full signal through the MLP.
        """
        # Lazy ST model (cached at ~/.cache/huggingface after feature_store.py)
        if not hasattr(self, '_st_model') or self._st_model is None:
            from sentence_transformers import SentenceTransformer
            st_name = self.cfg['models']['two_tower']['sentence_transformer_model']
            self._st_model = SentenceTransformer(st_name)
            logger.info(f'ST model loaded: {st_name}')

        text = f"{title} {'|'.join(genres)}"
        content_np = self._st_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

        genre_vec = np.zeros((1, len(GENRES)), dtype=np.float32)
        for g in genres:
            if g in GENRES:
                genre_vec[0, GENRES.index(g)] = 1.0

        item_id_t = torch.tensor([0], dtype=torch.long, device=self.device)
        content_t = torch.tensor(content_np, dtype=torch.float32, device=self.device)
        genre_t = torch.tensor(genre_vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            new_emb = self.two_tower.encode_items(item_id_t, content_t, genre_t)
            new_emb = new_emb.squeeze(0).cpu().numpy()

        # Similar existing items
        item_scores = self.tt_item_embs @ new_emb
        sim_ids = np.argsort(item_scores)[::-1][:n_similar]
        sim_scores = item_scores[sim_ids]

        # Top users by embedding closeness to this new item
        user_scores = self.tt_user_embs @ new_emb
        top_uids = np.argsort(user_scores)[::-1][:n_users]
        top_u_scores = user_scores[top_uids]

        # Demographic breakdown of top users vs dataset baseline
        gender_top = {'Male': 0, 'Female': 0}
        age_top = {v: 0 for v in AGE_LABELS.values()}
        for uid in top_uids:
            d = self.user_demo.get(int(uid), {'gender_enc': 0, 'age_enc': 0, 'occupation': 0})
            gender_top['Male' if d['gender_enc'] == 1 else 'Female'] += 1
            age_top[AGE_LABELS.get(int(d['age_enc']), '<18')] += 1

        gender_base = {'Male': 0, 'Female': 0}
        age_base = {v: 0 for v in AGE_LABELS.values()}
        for d in self.user_demo.values():
            gender_base['Male' if d['gender_enc'] == 1 else 'Female'] += 1
            age_base[AGE_LABELS.get(int(d['age_enc']), '<18')] += 1
        n_base = max(len(self.user_demo), 1)

        gender_bias = {
            g: {
                'top_pct': round(gender_top[g] / n_users, 3),
                'base_pct': round(gender_base[g] / n_base, 3),
            }
            for g in ['Male', 'Female']
        }
        age_bias = {
            label: {
                'top_pct': round(age_top.get(label, 0) / n_users, 3),
                'base_pct': round(age_base.get(label, 0) / n_base, 3),
            }
            for label in AGE_LABELS.values()
        }

        return {
            'similar_items': {
                'items': sim_ids.tolist(),
                'scores': sim_scores.tolist(),
            },
            'top_users': {
                'user_ids': top_uids.tolist(),
                'scores': top_u_scores.tolist(),
            },
            'gender_bias': gender_bias,
            'age_bias': age_bias,
            'input': {'title': title, 'genres': genres},
        }

    def recommend_cold_item(self, item_id: int, N: int = 10) -> dict:
        """Unchanged — Two-Tower item-item similarity via encode_items."""
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

    # -------------------------------------------------------------------------
    # Demographics helpers (Tab 1 + Tab 3)
    # -------------------------------------------------------------------------

    def trending_for_demo(self, gender_enc: int, age_enc: int, N: int = 10) -> list:
        """Most popular items among users matching gender_enc + age_enc."""
        from collections import Counter
        counts = Counter()
        for uid, demo in self.user_demo.items():
            if demo['gender_enc'] == gender_enc and demo['age_enc'] == age_enc:
                for item in self.user_history.get(uid, []):
                    counts[item] += 1
        if not counts:
            return self.popular_items[:N]
        return [item for item, _ in counts.most_common(N)]

    def two_tower_for_demo(self, gender_enc: int, age_enc: int, N: int = 10) -> list:
        """Two-Tower with demographics only — no history.
        user_id=0 (padding), empty history (zeros), demographics passed via demo_feats.
        """
        max_seq = self.cfg['models']['two_tower']['max_seq_len']
        uid_t = torch.tensor([0], dtype=torch.long, device=self.device)
        hist_t = torch.zeros(1, max_seq, dtype=torch.long, device=self.device)
        lens_t = torch.tensor([1], dtype=torch.long)  # GRU pack requires >=1
        demo_t = torch.tensor(
            [[float(gender_enc), float(age_enc), 0.0]],
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            u_emb = self.two_tower.encode_users(uid_t, hist_t, lens_t, demo_t)
            u_emb = u_emb.squeeze(0).cpu().numpy()
        scores = self.tt_item_embs @ u_emb
        return np.argsort(scores)[::-1][:N].tolist()

    def two_tower_with_demo(self, history: list, gender_enc: int,
                             age_enc: int, N: int = 10) -> list:
        """Two-Tower with interaction history AND explicit demographics."""
        max_seq = self.cfg['models']['two_tower']['max_seq_len']
        seq = history[-max_seq:]
        hist_len = max(len(seq), 1)
        padded = [0] * (max_seq - len(seq)) + seq
        hist_t = torch.tensor([padded], dtype=torch.long, device=self.device)
        lens_t = torch.tensor([hist_len], dtype=torch.long)
        uid_t = torch.tensor([0], dtype=torch.long, device=self.device)
        demo_t = torch.tensor(
            [[float(gender_enc), float(age_enc), 0.0]],
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            u_emb = self.two_tower.encode_users(uid_t, hist_t, lens_t, demo_t)
            u_emb = u_emb.squeeze(0).cpu().numpy()
        scores = self.tt_item_embs @ u_emb
        seen = [i for i in history if i < self.n_items]
        if seen:
            scores = scores.copy()
            scores[seen] = -np.inf
        return np.argsort(scores)[::-1][:N].tolist()

    def new_user_cold_start(self, gender_enc: int, age_enc: int, N: int = 10) -> dict:
        """Cold start for new user with demographics but no history.
        Returns trending for that demographic AND Two-Tower for that demographic.
        """
        return {
            'trending':   self.trending_for_demo(gender_enc, age_enc, N),
            'two_tower':  self.two_tower_for_demo(gender_enc, age_enc, N),
            'gender_enc': gender_enc,
            'age_enc':    age_enc,
        }

    # -------------------------------------------------------------------------
    # Homepage helpers (Tab 5 — Netflix-style)
    # -------------------------------------------------------------------------

    def get_user_top_genres(self, history: list, top_n: int = 2) -> list:
        """Top_n genre names by count from user's watch history."""
        if not history:
            return []
        valid = [i for i in history if i < self.n_items]
        if not valid:
            return []
        genre_counts = np.zeros(len(GENRES), dtype=np.float32)
        for iid in valid:
            genre_counts += self.item_genre[iid]
        top_idx = np.argsort(genre_counts)[::-1][:top_n]
        return [GENRES[i] for i in top_idx if genre_counts[i] > 0]

    def get_genre_rows_for_user(self, history: list,
                                 n_candidates: int = 200, top_n: int = 5) -> dict:
        """
        Two-Tower top n_candidates for user -> filter by top 2 genres from history.
        Returns {genre_name: [item_ids]} for up to top 2 genres.
        """
        top_genres = self.get_user_top_genres(history, top_n=2)
        if not top_genres:
            return {}
        tt_items = self._two_tower_for_user(None, history, n_candidates)
        result = {}
        for genre in top_genres:
            if genre not in GENRES:
                continue
            g_idx = GENRES.index(genre)
            filtered = [
                iid for iid in tt_items
                if iid < self.n_items and self.item_genre[iid][g_idx] > 0
            ][:top_n]
            if filtered:
                result[genre] = filtered
        return result

    def get_cold_genre_rows(self, gender_enc: int, age_enc: int,
                             top_n: int = 5) -> dict:
        """
        For cold users: filter popular_items by demographic genres.
        Row 'age_gender_1': top genre for age+gender
        Row 'age_gender_2': 2nd genre for age+gender
        Row 'gender_only':  top genre for gender alone (distinct from the first two).
        """
        ag_genre = np.zeros(len(GENRES), dtype=np.float32)
        for uid, demo in self.user_demo.items():
            if demo['gender_enc'] == gender_enc and demo['age_enc'] == age_enc:
                for iid in self.user_history.get(uid, []):
                    if iid < self.n_items:
                        ag_genre += self.item_genre[iid]
        ag_top = np.argsort(ag_genre)[::-1]
        ag_genre1 = GENRES[ag_top[0]] if ag_genre[ag_top[0]] > 0 else None
        ag_genre2 = GENRES[ag_top[1]] if len(ag_top) > 1 and ag_genre[ag_top[1]] > 0 else None

        g_genre = np.zeros(len(GENRES), dtype=np.float32)
        for uid, demo in self.user_demo.items():
            if demo['gender_enc'] == gender_enc:
                for iid in self.user_history.get(uid, []):
                    if iid < self.n_items:
                        g_genre += self.item_genre[iid]
        used = {ag_genre1, ag_genre2}
        g_top = [GENRES[i] for i in np.argsort(g_genre)[::-1] if GENRES[i] not in used]
        g_genre1 = g_top[0] if g_top else None

        def top_items_in_genre(genre, n):
            if genre is None:
                return []
            g_idx = GENRES.index(genre)
            return [
                iid for iid in self.popular_items
                if iid < self.n_items and self.item_genre[iid][g_idx] > 0
            ][:n]

        return {
            'age_gender_1': {'genre': ag_genre1, 'items': top_items_in_genre(ag_genre1, top_n)},
            'age_gender_2': {'genre': ag_genre2, 'items': top_items_in_genre(ag_genre2, top_n)},
            'gender_only':  {'genre': g_genre1,  'items': top_items_in_genre(g_genre1, top_n)},
        }

    def get_simulated_new_arrivals(self, N: int = 10) -> list:
        """Fixed-seed random sample of real movies — simulates newly added catalog items."""
        rng = np.random.default_rng(42)
        return rng.choice(self.n_items, size=N, replace=False).tolist()

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _tier1(self, uid, N):
        candidates = self.popular_items[:100]
        cscores = np.arange(100, 0, -1, dtype=np.float32)
        feat = build_features(uid, candidates, cscores,
                              self.user_demo, self.n_history_map,
                              self.user_genre, self.item_genre, self.item_pop)
        items = self.reranker.rerank(feat, k=N)
        return items, 1, 'popularity+reranker'

    def _tier2(self, uid, history, N, exclude_seen):
        excl = history if exclude_seen else []
        cf_ids, cf_sc = self.cf.retrieve(history, k=100, exclude_ids=excl)
        svd_ids, svd_sc = self.svd.retrieve(history, k=100, exclude_ids=excl)
        merged = {}
        for iid, sc in zip(cf_ids, cf_sc):
            merged[int(iid)] = float(sc)
        for iid, sc in zip(svd_ids, svd_sc):
            k = int(iid)
            if k not in merged or float(sc) > merged[k]:
                merged[k] = float(sc)
        cids = np.array(list(merged.keys()), dtype=np.int32)
        cscores = np.array(list(merged.values()), dtype=np.float32)
        feat = build_features(uid, cids, cscores,
                              self.user_demo, self.n_history_map,
                              self.user_genre, self.item_genre, self.item_pop)
        items = self.reranker.rerank(feat, k=N)
        return items, 2, 'cf+svd+reranker'

    def _tier3(self, history, N, exclude_seen):
        max_len = self.sasrec_max_len
        seq = [i + 1 for i in history[-max_len:]]
        padded = [0] * (max_len - len(seq)) + seq
        seq_t = torch.tensor([padded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.sasrec.score_all_items(seq_t).squeeze(0).cpu().numpy()
        if exclude_seen:
            seen = [i for i in history if i < self.n_items]
            logits[seen] = -np.inf
        items = np.argsort(logits)[::-1][:N].tolist()
        return items, 3, 'sasrec'

    def _sasrec_score(self, history, N):
        items, _, _ = self._tier3(history, N, exclude_seen=True)
        return items

    def _bert4rec_score(self, history, N):
        max_len = self.bert4rec_max_len
        seq = [i + 1 for i in history[-(max_len - 1):]]        # shift +1; leave 1 slot for MASK
        padded = [0] * (max_len - 1 - len(seq)) + seq + [self.bert4rec_mask]
        seq_t = torch.tensor([padded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.bert4rec.score_all_items(seq_t).squeeze(0).cpu().numpy()
        seen = [i for i in history if i < self.n_items]
        if seen:
            logits[seen] = -np.inf
        return np.argsort(logits)[::-1][:N].tolist()

    def _two_tower_for_user(self, user_id, history, N):
        """
        Known user (0 <= user_id < n_users): precomputed user_emb.npy.
        Unknown user: fold-in via user_tower with user_id=0 placeholder + zero demographics.
        """
        if user_id is not None and 0 <= user_id < len(self.tt_user_embs):
            u_emb = self.tt_user_embs[user_id]
        else:
            max_seq = self.cfg['models']['two_tower']['max_seq_len']
            seq = history[-max_seq:]
            hist_len = max(len(seq), 1)
            padded = [0] * (max_seq - len(seq)) + seq
            hist_t = torch.tensor([padded], dtype=torch.long, device=self.device)
            lens_t = torch.tensor([hist_len], dtype=torch.long)
            uid_t = torch.tensor([0], dtype=torch.long, device=self.device)
            demo_t = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                u_emb = self.two_tower.encode_users(
                    uid_t, hist_t, lens_t, demo_t,
                ).squeeze(0).cpu().numpy()

        scores = self.tt_item_embs @ u_emb
        seen = [i for i in history if i < self.n_items]
        if seen:
            scores = scores.copy()
            scores[seen] = -np.inf
        return np.argsort(scores)[::-1][:N].tolist()
