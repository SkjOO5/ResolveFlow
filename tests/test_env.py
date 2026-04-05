import pytest
from envs.environment import OpenSupportEnv
from envs.models import Action

def test_reset_creates_valid_observation():
    env = OpenSupportEnv()
    obs = env.reset("task_001_easy")
    assert obs.task_id == "task_001_easy"
    assert obs.step_count == 0
    assert not obs.done
    assert len(obs.available_actions) > 0

def test_step_logic():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    
    # valid test action
    res = env.step(Action(action_type="request_account_details", payload={}))
    assert res.reward.value != 0 # should be non-zero (tool reward or step penalty)
    assert not res.done
    assert "account" in res.observation.revealed_context

def test_grader():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="classify_ticket", payload={"label":"damaged_item"}))
    env.step(Action(action_type="set_priority", payload={"priority":"medium"}))
    env.step(Action(action_type="issue_refund", payload={"amount":0}))
    env.step(Action(action_type="draft_response", payload={"message":"refund apology keep item"}))
    res = env.step(Action(action_type="close_ticket", payload={}))
    
    assert res.done
    assert res.info["final_score"] == 1.0 # flawless execution

def test_invalid_action_penalized():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    
    res = env.step(Action(action_type="classify_ticket", payload={"label":"wrong"}))
    assert res.reward.value < 0 # heavily penalized

def test_max_steps_termination():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    # task has 10 max steps
    for _ in range(9):
        env.step(Action(action_type="request_return_policy", payload={}))
    
    res = env.step(Action(action_type="request_return_policy", payload={}))
    assert res.done
