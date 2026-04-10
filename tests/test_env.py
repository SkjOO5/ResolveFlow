import pytest
from fastapi.testclient import TestClient
from envs.environment import OpenSupportEnv
from envs.models import Action
from envs.scoring import strict_score

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
    """Test that a flawless performance yields a high score (strictly < 1.0)."""
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
    env.step(Action(action_type="set_priority", payload={"priority": "medium"}))
    env.step(Action(action_type="request_account_details", payload={}))
    env.step(Action(action_type="draft_response", payload={"message": "refund apology keep item"}))
    res = env.step(Action(action_type="issue_refund", payload={}))

    assert res.done
    score = res.info["final_score"]
    # Score should be high but strictly less than 1.0 (OpenEnv requirement)
    assert 0.9 < score < 1.0, f"Score should be high but open-interval: {score}"

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
    score = res.info["final_score"]
    assert 0 < score < 0.5, f"Policy breach should drop score below 0.5: {score}"

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

def test_score_strictly_open_interval_all_tasks():
    """
    CRITICAL: OpenEnv requires scores to be strictly in (0, 1).
    This test ensures NO task ever returns exactly 0.0 or 1.0.
    """
    for task_id in ["task_001_easy", "task_002_medium", "task_003_hard"]:
        env = OpenSupportEnv()
        env.reset(task_id)
        res = env.step(Action(action_type="close_ticket", payload={}))
        score = res.info.get("final_score", 0.0)
        
        # CRITICAL ASSERTIONS
        assert 0.0 < score < 1.0, f"Score MUST be in (0, 1), got {score} for {task_id}"
        assert score != 0.0, f"Score must NOT be exactly 0.0 for {task_id}"
        assert score != 1.0, f"Score must NOT be exactly 1.0 for {task_id}"

def test_all_breakdown_components_in_open_interval():
    """Test that all score breakdown dimensions are strictly in (0, 1)."""
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
    env.step(Action(action_type="set_priority", payload={"priority": "medium"}))
    env.step(Action(action_type="issue_refund", payload={}))
    
    state = env.current_state()
    breakdown = state.score_breakdown
    
    assert breakdown is not None, "Score breakdown must be present"
    for dimension, value in breakdown.items():
        assert 0.0 < value < 1.0, (
            f"Dimension '{dimension}' must be in (0, 1), got {value}"
        )

def test_strict_score_function():
    """Test the strict_score helper function."""
    # Boundary cases should map to interior values
    assert 0.0 < strict_score(0.0) < 1.0
    assert 0.0 < strict_score(1.0) < 1.0
    
    # Middle values should stay approximately middle
    middle = strict_score(0.5)
    assert 0.4 < middle < 0.6
    
    # Out-of-range values should be clamped and mapped
    assert 0.0 < strict_score(-10.0) < 1.0
    assert 0.0 < strict_score(10.0) < 1.0
    
    # Invalid inputs should return safe values
    assert 0.0 < strict_score(float('nan')) < 1.0
    assert 0.0 < strict_score(float('inf')) < 1.0

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
    # Tool usage dimension (20%) should be penalized, so score < high threshold
    assert 0 < score < 0.95, f"Missing tools should reduce score, got {score}"

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
    assert result["status"] == "healthy"

def test_reset_endpoint():
    from app import reset_env, ResetRequest
    import asyncio
    result = asyncio.run(reset_env(ResetRequest(task_id="task_001_easy")))
    assert result["task_id"] == "task_001_easy"
    assert result["step_count"] == 0
    assert result["done"] is False

def test_step_endpoint():
    from app import reset_env, step_env, ResetRequest
    import asyncio
    asyncio.run(reset_env(ResetRequest(task_id="task_001_easy")))
    data = asyncio.run(step_env(Action(action_type="request_account_details", payload={})))
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
    import asyncio
    asyncio.run(reset_env(ResetRequest(task_id="task_001_easy")))
    asyncio.run(step_env(Action(action_type="classify_ticket", payload={"label": "damaged_item"})))
    asyncio.run(step_env(Action(action_type="set_priority", payload={"priority": "medium"})))
    asyncio.run(step_env(Action(action_type="draft_response", payload={"message": "refund apology keep item"})))
    data = asyncio.run(step_env(Action(action_type="issue_refund", payload={})))
    assert data["done"] is True
    assert data["info"]["final_score"] is not None
