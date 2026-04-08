from envs.models import TaskDefinition, Rubric
from envs.datasets import ACCOUNTS_DB, ORDERS_DB, REFUND_HISTORY_DB, BILLING_HISTORY_DB, RETURN_POLICIES

EASY_TASK = TaskDefinition(
    task_id="task_001_easy",
    title="Damaged Item Refund",
    difficulty="easy",
    customer_message="Hi, I just received my order but the ceramic mug is completely shattered in the box. Can I get a refund?",
    visible_metadata={
        "customer_id": "CUST-1001",
        "order_id": "ORD-1A"
    },
    hidden_rubric=Rubric(
        correct_classification="damaged_item",
        correct_priority="medium",
        refund_eligible=True,
        replacement_eligible=True,
        store_credit_acceptable=True,
        escalation_required=False,
        required_tool_calls=[],
        valid_terminal_actions=["issue_refund", "offer_replacement", "close_ticket"],
        required_response_elements=["refund", "apology", "keep"],
        prohibited_response_elements=["return", "ship it back"]
    ),
    max_steps=10,
    internal_data_store={
        "account": ACCOUNTS_DB["CUST-1001"],
        "order": ORDERS_DB["ORD-1A"],
        "refund_history": REFUND_HISTORY_DB["CUST-1001"],
        "billing": BILLING_HISTORY_DB["CUST-1001"],
        "policy": RETURN_POLICIES["general"]
    }
)

MEDIUM_TASK = TaskDefinition(
    task_id="task_002_medium",
    title="Delayed Delivery",
    difficulty="medium",
    customer_message="Where is my order? It was supposed to be here last week and the tracking hasn't updated. I need this soon.",
    visible_metadata={
        "customer_id": "CUST-1002",
        "order_id": "ORD-2B"
    },
    hidden_rubric=Rubric(
        correct_classification="delayed_shipment",
        correct_priority="medium",
        refund_eligible=True, # Conditional (15%)
        replacement_eligible=False,
        store_credit_acceptable=True,
        escalation_required=False,
        required_tool_calls=["request_shipping_status", "request_return_policy"],
        valid_terminal_actions=["draft_response", "close_ticket", "issue_refund", "offer_store_credit"],
        required_response_elements=["delay", "apology", "15%"],
        prohibited_response_elements=["full refund", "cancel"]
    ),
    max_steps=15,
    internal_data_store={
        "account": ACCOUNTS_DB["CUST-1002"],
        "order": ORDERS_DB["ORD-2B"],
        "refund_history": REFUND_HISTORY_DB["CUST-1002"],
        "billing": BILLING_HISTORY_DB["CUST-1002"],
        "policy": RETURN_POLICIES["delayed_shipping"]
    }
)

HARD_TASK = TaskDefinition(
    task_id="task_003_hard",
    title="High-Value Suspected Fraud",
    difficulty="hard",
    customer_message="I didn't receive my watch!! The box was empty. Refund me immediately to my card or I'll chargeback.",
    visible_metadata={
        "customer_id": "CUST-2099",
        "order_id": "ORD-3C"
    },
    hidden_rubric=Rubric(
        correct_classification="missing_item",
        correct_priority="high",
        refund_eligible=False,
        replacement_eligible=False,
        store_credit_acceptable=False,
        escalation_required=True,
        required_tool_calls=["request_account_details", "request_refund_history", "request_billing_history", "request_return_policy"],
        valid_terminal_actions=["escalate_to_human"],
        required_response_elements=["escalate", "investigate", "team"],
        prohibited_response_elements=["refund", "replacement", "store credit"]
    ),
    max_steps=20,
    internal_data_store={
        "account": ACCOUNTS_DB["CUST-2099"],
        "order": ORDERS_DB["ORD-3C"],
        "refund_history": REFUND_HISTORY_DB["CUST-2099"],
        "billing": BILLING_HISTORY_DB["CUST-2099"],
        "policy": RETURN_POLICIES["fraud_protocol"]
    }
)

ALL_TASKS = {
    EASY_TASK.task_id: EASY_TASK,
    MEDIUM_TASK.task_id: MEDIUM_TASK,
    HARD_TASK.task_id: HARD_TASK
}

def get_task_by_id(task_id: str) -> TaskDefinition:
    return ALL_TASKS.get(task_id, EASY_TASK)

def get_task_by_difficulty(diff: str) -> TaskDefinition:
    for t in ALL_TASKS.values():
        if t.difficulty == diff:
            return t
    return EASY_TASK
