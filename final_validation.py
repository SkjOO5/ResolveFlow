"""
Final end-to-end validation simulating the actual OpenEnv validator.
This test verifies the complete submission workflow.
"""

import sys
sys.path.insert(0, '.')

print("\n" + "="*80)
print("FINAL END-TO-END OPENENV VALIDATOR SIMULATION")
print("="*80)

from app import app
from fastapi.testclient import TestClient
from envs.environment import OpenSupportEnv
from envs.models import Action

client = TestClient(app)

# === PHASE 1: Discovery ===
print("\n[PHASE 1] Task Discovery")
print("-" * 80)

response = client.get('/tasks')
tasks = response.json()

print(f"Total tasks found: {len(tasks)}")
assert len(tasks) >= 3, f"Need >= 3 tasks, found {len(tasks)}"

for i, task in enumerate(tasks, 1):
    print(f"\n  Task {i}: {task['id']}")
    print(f"    Title: {task['title']}")
    print(f"    Difficulty: {task['difficulty']}")
    print(f"    Max Steps: {task['max_steps']}")
    print(f"    Has Grader: {task.get('has_grader', False)}")
    assert task.get('has_grader') == True

print(f"\n✓ PHASE 1 PASSED: All {len(tasks)} tasks with graders discovered")

# === PHASE 2: Grading Validation ===
print("\n[PHASE 2] Grader Score Range Validation")
print("-" * 80)

violations = []
scores_collected = []

for task in tasks:
    task_id = task['id']
    
    # Run minimal grading scenario
    env = OpenSupportEnv()
    env.reset(task_id)
    res = env.step(Action(action_type="close_ticket", payload={}))
    
    if res.done:
        score = res.info.get('final_score')
        scores_collected.append((task_id, score))
        
        # Validate open interval
        if not (0.0 < score < 1.0):
            violations.append(f"{task_id}: Score {score} NOT in (0, 1)")
        if score == 0.0:
            violations.append(f"{task_id}: Score equals exactly 0.0")
        if score == 1.0:
            violations.append(f"{task_id}: Score equals exactly 1.0")
        
        status = "✓" if (0.0 < score < 1.0) else "✗"
        print(f"  {status} {task_id:20} score = {score:.8f}")

if violations:
    print("\n✗ PHASE 2 FAILED:")
    for v in violations:
        print(f"    - {v}")
    sys.exit(1)

print(f"\n✓ PHASE 2 PASSED: All {len(scores_collected)} task scores valid")

# === PHASE 3: Determinism ===
print("\n[PHASE 3] Deterministic Grading Validation")
print("-" * 80)

for task in tasks:
    scores = []
    for run in range(2):
        env = OpenSupportEnv()
        env.reset(task['id'])
        res = env.step(Action(action_type="close_ticket", payload={}))
        if res.done:
            scores.append(res.info.get('final_score'))
    
    if len(scores) == 2:
        if scores[0] == scores[1]:
            print(f"  ✓ {task['id']:20} deterministic: {scores[0]:.8f}")
        else:
            print(f"  ✗ {task['id']:20} NON-DETERMINISTIC: {scores[0]} vs {scores[1]}")
            sys.exit(1)

print(f"\n✓ PHASE 3 PASSED: All tasks produce deterministic scores")

# === PHASE 4: Breakdown Dimensions ===
print("\n[PHASE 4] Score Breakdown Dimension Validation")
print("-" * 80)

env = OpenSupportEnv()
env.reset("task_001_easy")
env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
env.step(Action(action_type="set_priority", payload={"priority": "medium"}))
env.step(Action(action_type="issue_refund", payload={}))

state = env.current_state()
breakdown = state.score_breakdown

print(f"\n  Task: task_001_easy")
print(f"  Final Score: {state.final_score:.8f}")
print(f"\n  Breakdown Dimensions:")

dimension_issues = []
for dim, value in breakdown.items():
    valid = 0.0 < value < 1.0
    status = "✓" if valid else "✗"
    print(f"    {status} {dim:20} = {value:.8f}")
    if not valid:
        dimension_issues.append(f"{dim} = {value}")

if dimension_issues:
    print(f"\n✗ PHASE 4 FAILED:")
    for issue in dimension_issues:
        print(f"    - {issue} not in (0, 1)")
    sys.exit(1)

print(f"\n✓ PHASE 4 PASSED: All {len(breakdown)} dimensions valid")

# === PHASE 5: API Health ===
print("\n[PHASE 5] API Endpoint Health Check")
print("-" * 80)

endpoints = [
    ("GET", "/health"),
    ("GET", "/tasks"),
    ("POST", "/reset"),
    ("GET", "/state"),
]

for method, endpoint in endpoints:
    if method == "GET":
        resp = client.get(endpoint)
    else:
        resp = client.post(endpoint, json={})
    
    status = "✓" if resp.status_code == 200 else "✗"
    print(f"  {status} {method:6} {endpoint:20} [HTTP {resp.status_code}]")
    assert resp.status_code == 200, f"Endpoint {method} {endpoint} failed"

print(f"\n✓ PHASE 5 PASSED: All API endpoints healthy")

# === FINAL REPORT ===
print("\n" + "="*80)
print("FINAL VALIDATION REPORT")
print("="*80)
print(f"\n✓ Task Registration:        PASSED (3+ tasks with graders)")
print(f"✓ Score Range:              PASSED (All scores in (0, 1))")
print(f"✓ Determinism:              PASSED (Reproducible outputs)")
print(f"✓ Breakdown Dimensions:     PASSED (All in (0, 1))")
print(f"✓ API Health:               PASSED (All endpoints operational)")
print(f"\n{'🎉 '*15}")
print("PROJECT IS VALID FOR OPENENV SUBMISSION")
print(f"{'🎉 '*15}")
print("="*80 + "\n")
