import requests


BASE = 'http://localhost:8000'


# Health
r = requests.get(f'{BASE}/health')
assert r.status_code == 200
print(f'Health: {r.json()}')


# Tier 1 — cold user
r = requests.post(f'{BASE}/recommend', json={'history': [], 'n': 5})
assert r.status_code == 200
body = r.json()
assert body['tier'] == 1 and len(body['items']) == 5
print(f'Tier 1 OK: latency={body["latency_ms"]}ms items={body["items"]}')


# Tier 2 — sparse user
r = requests.post(f'{BASE}/recommend', json={'history': [100, 200, 300], 'n': 5})
body = r.json()
assert body['tier'] == 2
print(f'Tier 2 OK: latency={body["latency_ms"]}ms items={body["items"]}')


# Tier 3 — warm user
r = requests.post(f'{BASE}/recommend', json={'history': list(range(20)), 'n': 5})
body = r.json()
assert body['tier'] == 3
print(f'Tier 3 OK: latency={body["latency_ms"]}ms items={body["items"]}')


# Similar items
r = requests.post(f'{BASE}/similar_items', json={'item_id': 42, 'n': 5})
body = r.json()
assert 42 not in body['items']
print(f'Similar items OK: {body["items"]}')


print('Section 10 API verified.')
