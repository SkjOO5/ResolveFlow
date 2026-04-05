from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

# -----------------
# Actions
# -----------------

ActionType = Literal[
    "classify_ticket",
    "set_priority",
    "request_account_details",
    "request_order_history",
    "request_shipping_status",
    "request_refund_history",
    "request_return_policy",
    "request_billing_history",
    "draft_response",
    "issue_refund",
    "offer_replacement",
    "offer_store_credit",
    "escalate_to_human",
    "close_ticket"
]

class Action(BaseModel):
    action_type: ActionType = Field(..., description="The type of action the agent wants to perform.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured payload for the action.")

# -----------------
# Environment Core Definitions
# -----------------

class Observation(BaseModel):
    task_id: str
    difficulty: str
    customer_message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    revealed_context: Dict[str, Any] = Field(default_factory=dict)
    available_actions: List[str] = Field(default_factory=list)
    step_count: int
    max_steps: int
    history_summary: List[str] = Field(default_factory=list)
    done: bool

class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    reason: str

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

# -----------------
# Internal Hidden State
# -----------------

class Rubric(BaseModel):
    correct_classification: str
    correct_priority: str
    refund_eligible: bool
    replacement_eligible: bool
    store_credit_acceptable: bool
    escalation_required: bool
    required_tool_calls: List[str]
    valid_terminal_actions: List[str]
    required_response_elements: List[str]
    prohibited_response_elements: List[str]

class TaskDefinition(BaseModel):
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    customer_message: str
    visible_metadata: Dict[str, Any]
    hidden_rubric: Rubric
    max_steps: int
    internal_data_store: Dict[str, Any] = Field(default_factory=dict, description="Hidden data retrieved via tools")

# -----------------
# Full State
# -----------------

class State(BaseModel):
    task_id: str
    difficulty: str
    step_count: int
    max_steps: int
    done: bool
    cumulative_reward: float
    last_reward: Reward | None = None
    action_history: List[Action] = Field(default_factory=list)
    revealed_context: Dict[str, Any] = Field(default_factory=dict)
    available_actions: List[str] = Field(default_factory=list)
    final_score: float | None = None
    score_breakdown: Dict[str, float] | None = None
    terminal_summary: str | None = None
    
    # Hidden tracking variables for reward & policy bounds
    classification_set: str | None = None
    priority_set: str | None = None
    response_drafted: str | None = None
