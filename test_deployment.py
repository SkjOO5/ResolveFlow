"""
Test that the deployment and server endpoints still work correctly.
"""

import sys
sys.path.insert(0, '.')

from app import app
from fastapi.testclient import TestClient

print("Testing deployment integrity...\n")

client = TestClient(app)

# Test /reset endpoint
print("[1] Testing POST /reset endpoint...")
response = client.post('/reset', json={"task_id": "task_001_easy"})
assert response.status_code == 200, f"Reset failed: {response.status_code}"
obs = response.json()
assert obs['task_id'] == 'task_001_easy'
print(f"    ✓ Reset works - got observation for task_001_easy")

# Test /health endpoint
print("\n[2] Testing /health endpoint...")
response = client.get('/health')
assert response.status_code == 200
health = response.json()
assert health['status'] == 'healthy'
print(f"    ✓ Health check passed: {health}")

# Test /state endpoint
print("\n[3] Testing /state endpoint...")
response = client.get('/state')
assert response.status_code == 200
state = response.json()
assert 'task_id' in state
print(f"    ✓ State endpoint works")

# Test /step endpoint
print("\n[4] Testing POST /step endpoint...")
response = client.post('/step', json={
    "action_type": "classify_ticket",
    "payload": {"label": "damaged_item"}
})
assert response.status_code == 200
step_result = response.json()
assert 'observation' in step_result
assert 'reward' in step_result
print(f"    ✓ Step endpoint works - reward={step_result['reward']['value']:.6f}")

print("\n✓ All deployment endpoints are functional!")
