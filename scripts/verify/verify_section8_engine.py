from config.config_loader import get_config
from inference.engine import TieredRecommendationEngine


cfg = get_config()
engine = TieredRecommendationEngine(cfg)


# Tier 1 — cold user
r = engine.recommend(None, [], N=10)
assert r['tier'] == 1 and r['model'] == 'popularity'
assert len(r['items']) == 10
print(f"Tier 1 OK: {r['model']}, latency={r['latency_ms']:.2f}ms")


# Tier 2 — sparse user (3 interactions)
r = engine.recommend(1, [100, 200, 300], N=10)
assert r['tier'] == 2 and r['model'] == 'svd_foldin'
assert len(r['items']) == 10
assert not any(i in [100, 200, 300] for i in r['items']), 'Seen items not excluded'
print(f"Tier 2 OK: {r['model']}, latency={r['latency_ms']:.2f}ms")


# Tier 3 — warm user (10 interactions)
r = engine.recommend(1, list(range(10)), N=10)
assert r['tier'] == 3 and r['model'] == 'sasrec'
assert len(r['items']) == 10
print(f"Tier 3 OK: {r['model']}, latency={r['latency_ms']:.2f}ms")


# Cold item retrieval
r = engine.recommend_cold_item(item_id=42, N=10)
assert r['model'] == 'two_tower'
assert 42 not in r['items'], 'Self not excluded'
print(f"Cold-item OK: {r['model']}, latency={r['latency_ms']:.2f}ms")


print('\nSection 8 engine verified.')
