import numpy as np
import pandas as pd
from config.config_loader import get_config
from inference.engine import TieredRecommendationEngine


def benchmark(engine, train_df, n_runs=100):
    results = {'tier1': [], 'tier2': [], 'tier3': []}

    # Tier 1 — cold users
    for _ in range(n_runs):
        r = engine.recommend(None, [], N=10)
        results['tier1'].append(r['latency_ms'])

    users = train_df['user_id'].unique()[:n_runs]
    user_hist = (
        train_df.sort_values('timestamp')
        .groupby('user_id', sort=False)['item_id']
        .apply(list).to_dict()
    )

    # Tier 2 — sparse users (3 interactions each)
    for uid in users:
        hist = user_hist.get(int(uid), [])[:3]
        r = engine.recommend(int(uid), hist, N=10)
        results['tier2'].append(r['latency_ms'])

    # Tier 3 — warm users (full history)
    for uid in users:
        hist = user_hist.get(int(uid), [])
        r = engine.recommend(int(uid), hist, N=10)
        results['tier3'].append(r['latency_ms'])

    print('Latency Benchmark (ms):')
    print(f'  {"Tier":<12} {"p50":>8} {"p95":>8} {"max":>8}')
    for tier, lats in results.items():
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        mx = max(lats)
        print(f'  {tier:<12} {p50:>8.2f} {p95:>8.2f} {mx:>8.2f}')
    return results


if __name__ == '__main__':
    cfg = get_config()
    engine = TieredRecommendationEngine(cfg)
    train_df = pd.read_parquet('data/processed/train.parquet')
    benchmark(engine, train_df)
