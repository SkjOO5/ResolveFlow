"""
Comprehensive validation test for OpenEnv submission.

This script tests all the requirements that the OpenEnv validator checks:
1. At least 3 tasks with graders are registered
2. Each task is discoverable via /tasks endpoint
3. Each task has a deterministic grader
4. All task scores are strictly in (0, 1)
5. No task ever returns exactly 0.0 or 1.0
"""

import sys
sys.path.insert(0, '.')

from app import app
from fastapi.testclient import TestClient
from envs.environment import OpenSupportEnv
from envs.models import Action

print("="*70)
print("OPENENV VALIDATOR SIMULATION")
print("="*70)

client = TestClient(app)

# === REQUIREMENT 1: At least 3 tasks with graders ===
print("\n[1] Checking task registration...")
response = client.get('/tasks')
assert response.status_code == 200, f"Failed to get /tasks: {response.status_code}"

tasks = response.json()
print(f"    Found {len(tasks)} tasks")
assert len(tasks) >= 3, f"Need at least 3 tasks, found {len(tasks)}"

task_ids = [t['id'] for t in tasks]
difficulties = [t['difficulty'] for t in tasks]

print(f"    Task IDs: {task_ids}")
print(f"    Difficulties: {difficulties}")

# Verify easy, medium, hard are present
assert 'easy' in difficulties, "Must have at least one 'easy' task"
assert 'medium' in difficulties, "Must have at least one 'medium' task"
assert 'hard' in difficulties, "Must have at least one 'hard' task"

print("    ✓ All difficulty levels represented")

# === REQUIREMENT 2 & 3: Task discoverability and grader attachment ===
print("\n[2] Checking task graders...")
for task in tasks:
    assert 'has_grader' in task, f"Task {task['id']} missing 'has_grader' field"
    assert task['has_grader'] == True, f"Task {task['id']} has no grader"
    assert 'grader_type' in task, f"Task {task['id']} missing 'grader_type'"
    assert task['grader_type'] == 'deterministic', f"Task {task['id']} grader not deterministic"
    print(f"    ✓ {task['id']}: {task['difficulty']} - {task['title']}")

# === REQUIREMENT 4 & 5: Score range validation ===
print("\n[3] Testing score range for all tasks (CRITICAL)...")
score_results = {}

for task_id in task_ids:
    # Run multiple scenarios to test different paths
    scenarios = [
        ("basic_close", [
            {"action_type": "close_ticket", "payload": {}}
        ]),
        ("classify_and_close", [
            {"action_type": "classify_ticket", "payload": {"label": "damaged_item"}},
            {"action_type": "close_ticket", "payload": {}}
        ]),
        ("full_path_easy", [
            {"action_type": "classify_ticket", "payload": {"label": "damaged_item"}},
            {"action_type": "set_priority", "payload": {"priority": "medium"}},
            {"action_type": "request_account_details", "payload": {}},
            {"action_type": "draft_response", "payload": {"message": "refund"}},
            {"action_type": "issue_refund", "payload": {}}
        ]) if task_id == "task_001_easy" else None,
    ]
    
    scenarios = [s for s in scenarios if s is not None]
    
    scores_for_task = []
    for scenario_name, actions in scenarios:
        env = OpenSupportEnv()
        env.reset(task_id)
        
        for action_spec in actions:
            res = env.step(Action(**action_spec))
            
            if res.done:
                score = res.info.get('final_score')
                scores_for_task.append(score)
                
                # CRITICAL: Verify open interval (0, 1)
                assert score is not None, f"{task_id}/{scenario_name}: No score returned"
                assert isinstance(score, float), f"{task_id}/{scenario_name}: Score not float: {type(score)}"
                assert 0.0 < score < 1.0, (
                    f"{task_id}/{scenario_name}: Score {score} NOT in (0, 1) - CRITICAL VIOLATION"
                )
                assert score != 0.0, f"{task_id}/{scenario_name}: Score is exactly 0.0 - VIOLATION"
                assert score != 1.0, f"{task_id}/{scenario_name}: Score is exactly 1.0 - VIOLATION"
                
                print(f"    ✓ {task_id:18} {scenario_name:20} score={score:.6f}")
    
    score_results[task_id] = scores_for_task

# === REQUIREMENT: Determinism ===
print("\n[4] Testing determinism...")
for task_id in task_ids[:1]:  # Test easy task
    env = OpenSupportEnv()
    env.reset(task_id)
    env.step(Action(action_type="close_ticket", payload={}))
    score1 = env.state.final_score
    
    env = OpenSupportEnv()
    env.reset(task_id)
    env.step(Action(action_type="close_ticket", payload={}))
    score2 = env.state.final_score
    
    assert score1 == score2, f"Scores not deterministic: {score1} vs {score2}"
    print(f"    ✓ {task_id}: Deterministic (score always {score1:.6f})")

# === SUMMARY ===
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print(f"✓ At least 3 tasks registered: {len(tasks)} tasks")
print(f"✓ All tasks have deterministic graders")
print(f"✓ All task scores are strictly in (0, 1)")
print(f"✓ No task returns exactly 0.0 or 1.0")
print(f"✓ Grader outputs are deterministic")
print("\n🎉 PROJECT PASSES OPENENV VALIDATION!")
print("="*70)
