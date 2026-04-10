import sys
sys.path.insert(0, '.')

from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Test /tasks endpoint
print('Testing /tasks endpoint...')
response = client.get('/tasks')
tasks = response.json()

print(f'Number of tasks: {len(tasks)}')
assert len(tasks) == 3, f'Must have at least 3 tasks, got {len(tasks)}'

for task in tasks:
    print(f"\nTask: {task['id']}")
    print(f"  Title: {task['title']}")
    print(f"  Difficulty: {task['difficulty']}")
    print(f"  Max steps: {task['max_steps']}")
    print(f"  Has grader: {task.get('has_grader', False)}")
    print(f"  Grader type: {task.get('grader_type', 'N/A')}")
    
    assert task['has_grader'] == True, f'Task {task["id"]} must have grader'
    assert task['grader_type'] == 'deterministic', f'Task {task["id"]} must have deterministic grader'

print('\n✓ All tasks are properly registered with graders!')
