"""
PREREQ: FastAPI server must be running on localhost:8000.
Start with: python -m api.main  (waits ~15s for engine to load)
"""
import httpx


BASE = 'http://localhost:8000'


def test_health():
    r = httpx.get(f'{BASE}/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert 'models' in body
    assert body['models']['tier3'] == 'sasrec'


def test_recommend_cold_user():
    r = httpx.post(f'{BASE}/recommend', json={'history': [], 'n': 10})
    assert r.status_code == 200
    body = r.json()
    assert body['tier'] == 1
    assert body['model'] == 'popularity'
    assert len(body['items']) == 10
    assert body['latency_ms'] >= 0


def test_recommend_sparse_user():
    r = httpx.post(f'{BASE}/recommend', json={'history': [100, 200, 300], 'n': 5})
    assert r.status_code == 200
    body = r.json()
    assert body['tier'] == 2
    assert body['model'] == 'svd_foldin'
    assert len(body['items']) == 5
    assert not any(i in [100, 200, 300] for i in body['items'])


def test_recommend_warm_user():
    r = httpx.post(f'{BASE}/recommend', json={'history': list(range(10)), 'n': 10})
    assert r.status_code == 200
    body = r.json()
    assert body['tier'] == 3
    assert body['model'] == 'sasrec'
    assert len(body['items']) == 10


def test_recommend_default_n():
    r = httpx.post(f'{BASE}/recommend', json={'history': []})
    assert r.status_code == 200
    assert len(r.json()['items']) == 10


def test_recommend_invalid_n():
    r = httpx.post(f'{BASE}/recommend', json={'history': [], 'n': 999})
    assert r.status_code == 422  # Pydantic validation: n max is 100


def test_similar_items():
    r = httpx.post(f'{BASE}/similar_items', json={'item_id': 42, 'n': 10})
    assert r.status_code == 200
    body = r.json()
    assert body['model'] == 'two_tower'
    assert len(body['items']) == 10
    assert 42 not in body['items']


def test_similar_items_default_n():
    r = httpx.post(f'{BASE}/similar_items', json={'item_id': 0})
    assert r.status_code == 200
    assert len(r.json()['items']) == 10
