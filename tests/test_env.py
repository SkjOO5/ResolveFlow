"""
tests/test_env.py

Full test suite:
  - Environment unit tests
  - Score strict open-interval validation (CRITICAL for OpenEnv)
  - API route integration tests
  - /grade endpoint tests
  - Determinism tests
"""
import pytest
import asyncio
from envs.environment import OpenSupportEnv
from envs.models import Action
from envs.scoring import strict_score, SCORE_MIN, SCORE_MAX

# ── strict_score unit tests ───────────────────────────────────────────────────

def test_strict_score_boundaries():
    """Boundary inputs must map to SCORE_MIN and SCORE_MAX, never 0.0 or 1.0."""
    assert strict_score(0.0) == SCORE_MIN
    assert strict_score(1.0) == SCORE_MAX
    assert strict_score(0.0) > 0.0
    assert strict_score(1.0) < 1.0

def test_strict_score_midpoint():
    """Midpoint should stay near 0.5."""
    mid = strict_score(0.5)
    assert 0.4 < mid < 0.6

def test_strict_score_out_of_range():
    """Out-of-range values must be clamped and stay in open interval."""
    assert SCORE_MIN <= strict_score(-100.0) <= SCORE_MAX
    assert SCORE_MIN <= strict_score(100.0) <= SCORE_MAX
    assert strict_score(-100.0) > 0.0
    assert strict_score(100.0) < 1.0

def test_strict_score_nan_and_inf():
    """NaN and inf must return safe mid-range fallback."""
    assert strict_score(float('nan')) == 0.5
    assert strict_score(float('inf')) == 0.5
    assert strict_score(float('-inf')) == 0.5

def test_strict_score_never_zero_or_one():
    """strict_score must NEVER return exactly 0.0 or 1.0 for any float input."""
    test_values = [0.0, 1.0, -1.0, 2.0, -0.0, 0.0001, 0.9999, 0.5, 0.25, 0.75,
                   float('nan'), float('inf'), float('-inf')]
    for val in test_values:
        result = strict_score(val)
        assert result != 0.0, f"strict_score({val}) returned exactly 0.0"
        assert result != 1.0, f"strict_score({val}) returned exactly 1.0"
        assert 0.0 < result < 1.0, f"strict_score({val}) = {result} not in (0, 1)"

# ── Environment unit tests ────────────────────────────────────────────────────

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
    """A flawless performance yields a high score — strictly < 1.0 (open interval)."""
    env = OpenSupportEnv()
    env.reset("task_001_easy")
    env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
    env.step(Action(action_type="set_priority", payload={"priority": "medium"}))
    env.step(Action(action_type="request_account_details", payload={}))
    env.step(Action(action_type="draft_response", payload={"message": "refund apology keep item"}))
    res = env.step(Action(action_type="issue_refund", payload={}))

    assert res.done
    score = res.info["final_score"]
    assert 0.0 < score < 1.0, f"Score must be in open interval (0, 1): {score}"
    assert score != 0.0, f"Score must not be exactly 0.0: {score}"
    assert score != 1.0, f"Score must not be exactly 1.0: {score}"
    assert score >= SCORE_MIN, f"Score below SCORE_MIN: {score}"
    assert score <= SCORE_MAX, f"Score above SCORE_MAX: {score}"
    assert score > 0.75, f"Flawless performance should score high: {score}"

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
    assert 0.0 < score < 1.0, f"Policy breach score must still be in (0,1): {score}"
    assert score < 0.5, f"Policy breach should drop score below 0.5: {score}"

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

# ── CRITICAL: Score open-interval validation for all tasks ────────────────────

def test_score_strictly_open_interval_all_tasks():
    """
    CRITICAL: OpenEnv requires scores to be strictly in (0, 1).
    Tests both minimal paths and richer paths for all 3 task difficulties.
    """
    for task_id in ["task_001_easy", "task_002_medium", "task_003_hard"]:
        env = OpenSupportEnv()
        env.reset(task_id)
        res = env.step(Action(action_type="close_ticket", payload={}))
        score = res.info.get("final_score", 0.0)

        assert isinstance(score, float), f"Score must be float for {task_id}, got {type(score)}"
        assert 0.0 < score < 1.0, f"Score MUST be in (0, 1), got {score} for {task_id}"
        assert score != 0.0, f"Score must NOT be exactly 0.0 for {task_id}"
        assert score != 1.0, f"Score must NOT be exactly 1.0 for {task_id}"

def test_score_open_interval_multiple_scenarios():
    """Test open-interval rule under success, failure, and partial-completion paths."""
    scenarios = [
        # (task_id, actions_list, description)
        ("task_001_easy", [
            ("close_ticket", {})
        ], "cold_close"),
        ("task_001_easy", [
            ("classify_ticket", {"label": "damaged_item"}),
            ("close_ticket", {})
        ], "classify_then_close"),
        ("task_001_easy", [
            ("classify_ticket", {"label": "damaged_item"}),
            ("set_priority", {"priority": "medium"}),
            ("request_account_details", {}),
            ("draft_response", {"message": "refund apology keep item"}),
            ("issue_refund", {}),
        ], "full_success_path"),
        ("task_002_medium", [
            ("close_ticket", {})
        ], "medium_cold_close"),
        ("task_002_medium", [
            ("classify_ticket", {"label": "delayed_shipment"}),
            ("request_shipping_status", {}),
            ("request_return_policy", {}),
            ("draft_response", {"message": "sorry for the delay, here is 15% store credit"}),
            ("offer_store_credit", {}),
        ], "medium_success"),
        ("task_003_hard", [
            ("close_ticket", {})
        ], "hard_cold_close"),
        ("task_003_hard", [
            ("classify_ticket", {"label": "missing_item"}),
            ("set_priority", {"priority": "high"}),
            ("request_account_details", {}),
            ("request_refund_history", {}),
            ("request_billing_history", {}),
            ("request_return_policy", {}),
            ("draft_response", {"message": "we must escalate to the team to investigate"}),
            ("escalate_to_human", {"team": "fraud"}),
        ], "hard_success"),
        # Edge case: policy violation (refund on fraud task)
        ("task_003_hard", [
            ("issue_refund", {"amount": 899.99})
        ], "hard_policy_violation"),
    ]

    for task_id, actions, description in scenarios:
        env = OpenSupportEnv()
        env.reset(task_id)
        score = None
        for action_type, payload in actions:
            res = env.step(Action(action_type=action_type, payload=payload))
            if res.done:
                score = res.info.get("final_score")
                break
        # If episode didn't complete naturally, force close
        if score is None:
            res = env.step(Action(action_type="close_ticket", payload={}))
            score = res.info.get("final_score", 0.5)

        assert score is not None, f"{task_id}/{description}: No score returned"
        assert isinstance(score, float), f"{task_id}/{description}: Score not float: {type(score)}"
        assert 0.0 < score < 1.0, (
            f"CRITICAL: {task_id}/{description}: score {score} NOT in (0, 1)"
        )
        assert score != 0.0, f"{task_id}/{description}: Score is exactly 0.0 — VIOLATION"
        assert score != 1.0, f"{task_id}/{description}: Score is exactly 1.0 — VIOLATION"

def test_all_breakdown_components_in_open_interval():
    """Every score breakdown dimension must be strictly in (0, 1)."""
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
        assert value != 0.0, f"Dimension '{dimension}' is exactly 0.0 — VIOLATION"
        assert value != 1.0, f"Dimension '{dimension}' is exactly 1.0 — VIOLATION"

def test_at_least_3_tasks_registered():
    """CRITICAL: At least 3 tasks must be registered and accessible."""
    from envs.tasks import ALL_TASKS
    assert len(ALL_TASKS) >= 3, f"Need at least 3 tasks, got {len(ALL_TASKS)}"

def test_all_difficulties_present():
    """Must have easy, medium, and hard tasks."""
    from envs.tasks import ALL_TASKS
    difficulties = {t.difficulty for t in ALL_TASKS.values()}
    assert "easy" in difficulties, "Missing 'easy' task"
    assert "medium" in difficulties, "Missing 'medium' task"
    assert "hard" in difficulties, "Missing 'hard' task"

def test_all_tasks_have_graders():
    """Every task in ALL_TASKS must be gradeable (Grader.grade must not raise)."""
    from envs.tasks import ALL_TASKS
    from envs.graders import Grader

    for task_id, task in ALL_TASKS.items():
        env = OpenSupportEnv()
        env.reset(task_id)
        env.step(Action(action_type="close_ticket", payload={}))
        state = env.current_state()
        # Grader.grade must succeed and return valid score
        score, breakdown, summary, audit = Grader.grade(state, task)
        assert isinstance(score, float), f"Grader.grade for {task_id} must return float"
        assert 0.0 < score < 1.0, f"Grader.grade for {task_id} returned out-of-range: {score}"

# ── Determinism tests ─────────────────────────────────────────────────────────

def test_determinism():
    """Same task + same actions must produce identical outcomes across resets."""
    scores = []
    for _ in range(3):
        env = OpenSupportEnv()
        env.reset("task_001_easy")
        env.step(Action(action_type="classify_ticket", payload={"label": "damaged_item"}))
        env.step(Action(action_type="set_priority", payload={"priority": "medium"}))
        env.step(Action(action_type="draft_response", payload={"message": "refund apology keep item"}))
        res = env.step(Action(action_type="issue_refund", payload={}))
        scores.append(res.info["final_score"])
    assert scores[0] == scores[1] == scores[2], f"Environment is not deterministic: {scores}"

def test_determinism_all_tasks():
    """Determinism must hold for all three task difficulties."""
    for task_id in ["task_001_easy", "task_002_medium", "task_003_hard"]:
        score_a = _run_minimal(task_id)
        score_b = _run_minimal(task_id)
        assert score_a == score_b, (
            f"Non-deterministic scores for {task_id}: {score_a} vs {score_b}"
        )

def _run_minimal(task_id: str) -> float:
    """Run a minimal (cold close) episode and return the final score."""
    env = OpenSupportEnv()
    env.reset(task_id)
    res = env.step(Action(action_type="close_ticket", payload={}))
    return res.info.get("final_score", 0.5)

# ── API integration tests ─────────────────────────────────────────────────────

def test_health_endpoint():
    from app import health_check
    result = health_check()
    assert result["status"] == "healthy"

def test_reset_endpoint():
    from app import reset_env, ResetRequest
    result = asyncio.run(reset_env(ResetRequest(task_id="task_001_easy")))
    assert result["task_id"] == "task_001_easy"
    assert result["step_count"] == 0
    assert result["done"] is False

def test_step_endpoint():
    from app import reset_env, step_env, ResetRequest
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

def test_tasks_endpoint_returns_3_tasks():
    """CRITICAL: /tasks endpoint must return >= 3 tasks each with a grader field."""
    from app import list_tasks
    tasks = asyncio.run(list_tasks())
    assert len(tasks) >= 3, f"Need at least 3 tasks, got {len(tasks)}"
    for task in tasks:
        assert "grader" in task, f"Task {task.get('id')} missing 'grader' field"
        assert task["grader"]["type"] == "deterministic"
        assert task["grader"]["score_range"][0] > 0.0, "score_range min must be > 0"
        assert task["grader"]["score_range"][1] < 1.0, "score_range max must be < 1"
        assert task.get("has_grader") is True

def test_grade_endpoint():
    """POST /grade must return a valid score strictly in (0, 1)."""
    from app import grade_task
    result = asyncio.run(grade_task({"task_id": "task_001_easy"}))
    assert "score" in result
    score = result["score"]
    assert isinstance(score, float), f"Grade score must be float, got {type(score)}"
    assert 0.0 < score < 1.0, f"Grade score must be in (0, 1): {score}"
    assert score != 0.0, "Grade score must not be exactly 0.0"
    assert score != 1.0, "Grade score must not be exactly 1.0"
    assert result["open_interval"] is True
    assert result["score_min"] == SCORE_MIN
    assert result["score_max"] == SCORE_MAX

def test_full_episode_via_api():
    from app import reset_env, step_env, ResetRequest
    asyncio.run(reset_env(ResetRequest(task_id="task_001_easy")))
    asyncio.run(step_env(Action(action_type="classify_ticket", payload={"label": "damaged_item"})))
    asyncio.run(step_env(Action(action_type="set_priority", payload={"priority": "medium"})))
    asyncio.run(step_env(Action(action_type="draft_response", payload={"message": "refund apology keep item"})))
    data = asyncio.run(step_env(Action(action_type="issue_refund", payload={})))
    assert data["done"] is True
    score = data["info"]["final_score"]
    assert score is not None
    assert 0.0 < score < 1.0, f"Final score out of range: {score}"

# ── Episode audit tests ───────────────────────────────────────────────────────

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
    env.step(Action(action_type="classify_ticket", payload={"label": "missing_item"}))
    res = env.step(Action(action_type="escalate_to_human", payload={"team": "fraud"}))
    score = res.info.get("final_score", 0.0)
    assert 0.0 < score < 1.0, f"Score must be in (0,1): {score}"
    assert score < 0.95, f"Missing tools should reduce score below SCORE_MAX: {score}"
