import pytest
from envs.environment import OpenSupportEnv
from envs.models import Action

def test_reset_creates_valid_observation():
    env = OpenSupportEnv()
    obs = env.reset("task_001_easy")
    assert obs.task_id == "task_001_easy"
    assert obs.step_count == 0
    assert not obs.done
    assert "classify_ticket" in obs.available_actions

def test_step_logic_and_rewards():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    
    # Missing valid classification should be penalized or correctly rewarded if accurate
    res = env.step(Action(action_type="classify_ticket", payload={"label":"damaged_item"}))
    assert res.reward.components.get("correct_classification", 0) > 0
    
    res2 = env.step(Action(action_type="request_account_details", payload={}))
    assert "account" in res2.observation.revealed_context
    assert res2.reward.components.get("tool_lookup", 0) > 0

def test_grader_flawless():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="classify_ticket", payload={"label":"damaged_item"}))
    env.step(Action(action_type="set_priority", payload={"priority":"medium"}))
    env.step(Action(action_type="request_account_details", payload={})) # Not required but safe
    env.step(Action(action_type="draft_response", payload={"message":"refund apology keep item"}))
    res = env.step(Action(action_type="issue_refund", payload={}))
    
    assert res.done
    assert res.info["final_score"] == 1.0 # flawless execution

def test_invalid_action_penalized():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    
    res = env.step(Action(action_type="classify_ticket", payload={"label":"wrong"}))
    assert res.reward.components.get("invalid_classification") is not None

def test_policy_breach_drops_score():
    env = OpenSupportEnv()
    env.reset("task_003_hard") # Hard task requires escalation, prohibits refund
    
    # Try to shortcut refund
    res = env.step(Action(action_type="issue_refund", payload={}))
    assert res.done
    assert res.info["final_score"] < 0.5 # Should be heavily penalized
