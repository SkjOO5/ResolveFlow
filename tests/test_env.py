import pytest
from fastapi.testclient import TestClient
from envs.environment import OpenSupportEnv
from envs.models import Action

# ── Backend unit tests ────────────────────────────────────────────────────────

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

    res = env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
    assert res.reward.components.get("correct_classification", 0) > 0

    res2 = env.step(Action(action_type="request_account_details", payload={}))
    assert "account" in res2.observation.revealed_context
    assert res2.reward.components.get("tool_lookup", 0) > 0

def test_grader_flawless():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
    env.step(Action(action_type="set_priority", payload={"priority": "medium"}))
    env.step(Action(action_type="request_account_details", payload={}))
    env.step(Action(action_type="draft_response", payload={"message": "refund apology keep item"}))
    res = env.step(Action(action_type="issue_refund", payload={}))

    assert res.done
    assert res.info["final_score"] == 1.0

def test_invalid_action_penalized():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    res = env.step(Action(action_type="classify_ticket", payload={"label": "wrong"}))
    assert res.reward.components.get("invalid_classification") is not None

def test_policy_breach_drops_score():
    env = OpenSupportEnv()
    env.reset("task_003_hard")
    res = env.step(Action(action_type="issue_refund", payload={}))
    assert res.done
    assert res.info["final_score"] < 0.5

def test_action_history_records_details():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    res = env.step(Action(action_type="request_account_details", payload={}))

    state = env.current_state()
    assert len(state.action_history) == 1
    assert state.action_history[0].step == 1
    assert state.action_history[0].action.action_type == "request_account_details"
    assert state.action_history[0].reward == res.reward.value
    assert state.action_history[0].result_summary == res.reward.reason

def test_cumulative_reward():
    env = OpenSupportEnv()
    env.reset("task_001_easy")

    r1 = env.step(Action(action_type="request_account_details", payload={}))
    r2 = env.step(Action(action_type="request_order_history", payload={}))

    state = env.current_state()
    assert state.step_count == 2
    assert abs(state.cumulative_reward - (r1.reward.value + r2.reward.value)) < 1e-4

def test_terminal_state_completeness():
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="issue_refund", payload={}))
    state = env.current_state()

    assert state.done is True
    assert state.final_score is not None
    assert state.score_breakdown is not None
    assert state.terminal_summary is not None

def test_score_bounds_all_tasks():
    """Score must always remain in [0.0, 1.0] for all tasks."""
    for task_id in ["task_001_easy", "task_002_medium", "task_003_hard"]:
        env = OpenSupportEnv()
        env.reset(task_id)
        res = env.step(Action(action_type="close_ticket", payload={}))
        score = res.info.get("final_score", 0.0)
        assert 0.0 <= score <= 1.0, f"Score out of bounds for {task_id}: {score}"

def test_episode_audit_generated():
    """Episode audit should be present at terminal state."""
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="issue_refund", payload={}))
    state = env.current_state()

    assert state.episode_audit is not None
    assert len(state.episode_audit) > 0
    assert all(isinstance(s, str) for s in state.episode_audit)

def test_reward_components_present():
    """Reward components must be a non-empty dict after a step."""
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    res = env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
    assert isinstance(res.reward.components, dict)
    assert len(res.reward.components) > 0

def test_hard_task_requires_tools():
    """Hard task grader penalizes missing tool lookups."""
    env = OpenSupportEnv()
    env.reset("task_003_hard")
    # Skip all required tools and escalate immediately
    env.step(Action(action_type="classify_ticket", payload={"label": "missing_item"}))
    res = env.step(Action(action_type="escalate_to_human", payload={"team": "fraud"}))
    score = res.info.get("final_score", 0.0)
    # Tool usage dimension (20%) should be 0.0, so score < 1.0
    assert score < 1.0

def test_determinism():
    """Same task must produce identical outcomes across resets."""
    scores = []
    for _ in range(2):
        env = OpenSupportEnv()
        env.reset("task_001_easy")
        env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
        env.step(Action(action_type="set_priority", payload={"priority": "medium"}))
        env.step(Action(action_type="draft_response", payload={"message": "refund apology keep item"}))
        res = env.step(Action(action_type="issue_refund", payload={}))
        scores.append(res.info["final_score"])
    assert scores[0] == scores[1], "Environment is not deterministic"

# ── API integration tests ─────────────────────────────────────────────────────

# ── API integration tests (direct route calls) ───────────────────────────────

def test_health_endpoint():
    from app import health_check
    result = health_check()
    assert result["status"] == "ok"

def test_reset_endpoint():
    from app import reset_env, ResetRequest
    result = reset_env(ResetRequest(task_id="task_001_easy"))
    assert result["task_id"] == "task_001_easy"
    assert result["step_count"] == 0
    assert result["done"] is False

def test_step_endpoint():
    from app import reset_env, step_env, ResetRequest
    reset_env(ResetRequest(task_id="task_001_easy"))
    data = step_env(Action(action_type="request_account_details", payload={}))
    assert "observation" in data
    assert "reward" in data
    assert "done" in data

def test_invalid_action_raises():
    """Invalid action_type raises Pydantic ValidationError before hitting the route."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        Action(action_type="nonexistent_action", payload={})

def test_full_episode_via_api():
    from app import reset_env, step_env, ResetRequest
    reset_env(ResetRequest(task_id="task_001_easy"))
    step_env(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
    step_env(Action(action_type="set_priority", payload={"priority": "medium"}))
    step_env(Action(action_type="draft_response", payload={"message": "refund apology keep item"}))
    data = step_env(Action(action_type="issue_refund", payload={}))
    assert data["done"] is True
    assert data["info"]["final_score"] is not None
