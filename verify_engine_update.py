from config.config_loader import get_config
from inference.engine import TieredRecommendationEngine


cfg = get_config()
engine = TieredRecommendationEngine(cfg)
print('Engine loaded.')


# Tier 1 — 0 history
r = engine.recommend(user_id=None, history=[], N=10)
assert r['tier'] == 1
assert len(r['items']) == 10
assert 'reranker' in r['model']
print(f'Tier 1 OK: {r["model"]}, latency={r["latency_ms"]:.1f}ms')


# Tier 2 — sparse
r = engine.recommend(user_id=42, history=[50, 296, 527], N=10)
assert r['tier'] == 2
assert len(r['items']) == 10
assert not any(i in [50, 296, 527] for i in r['items'])
print(f'Tier 2 OK: {r["model"]}, latency={r["latency_ms"]:.1f}ms')


# Tier 3 — warm
history = [50, 296, 527, 858, 1197, 1240, 1265, 318, 593, 608]
r = engine.recommend(user_id=42, history=history, N=10)
assert r['tier'] == 3
assert r['model'] == 'sasrec'
print(f'Tier 3 OK: {r["model"]}, latency={r["latency_ms"]:.1f}ms')


# recommend_all_models
all_r = engine.recommend_all_models(user_id=42, history=history[:3], N=10)
for key in ['cf', 'svd', 'two_tower', 'sasrec', 'bert4rec', 'reranked']:
    assert key in all_r, f'Missing key: {key}'
    assert len(all_r[key]['items']) == 10, f'{key}: expected 10 items'
print(f'recommend_all_models OK: tier={all_r["tier"]}')


# item_similarity
sim = engine.item_similarity(item_id=50, N=10)
for model in ['cf', 'svd', 'two_tower']:
    assert model in sim
    assert len(sim[model]['items']) > 0
    assert 50 not in sim[model]['items'], f'{model}: item_id leaked into similarity'
print(f'item_similarity OK: CF={len(sim["cf"]["items"])} SVD={len(sim["svd"]["items"])} TT={len(sim["two_tower"]["items"])}')


# Cold item unchanged
c = engine.recommend_cold_item(item_id=50, N=10)
assert len(c['items']) == 10
print(f'recommend_cold_item OK: latency={c["latency_ms"]:.1f}ms')


print('All engine update checks passed.')
