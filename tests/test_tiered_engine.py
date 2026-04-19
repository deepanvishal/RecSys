from unittest.mock import MagicMock, patch
from inference.engine import TieredRecommendationEngine


def _bare_engine():
    """Construct a TieredRecommendationEngine without running __init__ (skip model loads)."""
    with patch.object(TieredRecommendationEngine, '__init__', return_value=None):
        engine = TieredRecommendationEngine.__new__(TieredRecommendationEngine)
    engine.TIER3_THRESHOLD = 5
    return engine


def test_tier1_cold_user():
    engine = _bare_engine()
    engine.popular_items = list(range(100))
    items, tier, model = engine._tier1(10)
    assert tier == 1
    assert model == 'popularity'
    assert len(items) == 10
    assert items == list(range(10))


def test_recommend_routes_tier1():
    engine = _bare_engine()
    engine.popular_items = list(range(100))
    engine._tier1 = MagicMock(return_value=(list(range(10)), 1, 'popularity'))
    engine._tier2 = MagicMock()
    engine._tier3 = MagicMock()
    result = engine.recommend(user_id=None, history=[], N=10)
    engine._tier1.assert_called_once()
    engine._tier2.assert_not_called()
    engine._tier3.assert_not_called()
    assert result['tier'] == 1


def test_recommend_routes_tier2():
    engine = _bare_engine()
    engine._tier1 = MagicMock()
    engine._tier2 = MagicMock(return_value=(list(range(10)), 2, 'svd_foldin'))
    engine._tier3 = MagicMock()
    result = engine.recommend(user_id=1, history=[10, 20, 30], N=10)
    engine._tier1.assert_not_called()
    engine._tier2.assert_called_once()
    engine._tier3.assert_not_called()
    assert result['tier'] == 2


def test_recommend_routes_tier3():
    engine = _bare_engine()
    engine._tier1 = MagicMock()
    engine._tier2 = MagicMock()
    engine._tier3 = MagicMock(return_value=(list(range(10)), 3, 'sasrec'))
    result = engine.recommend(user_id=1, history=list(range(10)), N=10)
    engine._tier1.assert_not_called()
    engine._tier2.assert_not_called()
    engine._tier3.assert_called_once()
    assert result['tier'] == 3


def test_tier_threshold_boundary():
    """Exactly 5 interactions → Tier 2 (boundary belongs to Tier 2)."""
    engine = _bare_engine()
    engine._tier1 = MagicMock()
    engine._tier2 = MagicMock(return_value=(list(range(10)), 2, 'svd_foldin'))
    engine._tier3 = MagicMock()
    result = engine.recommend(user_id=1, history=list(range(5)), N=10)
    engine._tier2.assert_called_once()
    engine._tier3.assert_not_called()


def test_recommend_returns_n_items():
    engine = _bare_engine()
    engine._tier1 = MagicMock(return_value=(list(range(5)), 1, 'popularity'))
    result = engine.recommend(user_id=None, history=[], N=5)
    assert len(result['items']) == 5
    assert 'latency_ms' in result
    assert 'n_history' in result
