# ResolveFlow / OpenSupportEnv

A real-world customer support ticket triage and resolution benchmark for evaluating autonomous AI agents, built on the **OpenEnv** specification.

## Overview

**ResolveFlow** implements the `OpenSupportEnv` environment, where an AI agent acts as a customer support representative. 
It must accurately triage customer tickets, safely use internal API tools to gather context (order history, account details), classify the problem, execute the correct resolution (e.g., refund vs. escalation), and draft a policy-compliant response.

This environment evaluates agents on **complex multi-step reasoning, tool usage, policy alignment, and operational efficiency.**

### Why This Environment Matters
- **Real-World Utility**: It directly simulates a massive multi-billion dollar real-world challenge: automated Tier-1 support orchestration.
- **Deterministic Evaluation**: Each task has a hidden mathematical rubric scoring classification, prioritization, resolution correctness, tool utilization, and response compliance in `[0.0, 1.0]`.
- **Safe & Bounded Action Space**: Prevents "gaming" the benchmark through explicit Pydantic schema validation.

## Environment Mechanics

### Observation Space
The agent receives a strict, typed JSON view containing:
- `customer_message`: The natural language issue reported by the customer.
- `metadata`: Initial visible context (e.g. `customer_id`).
- `revealed_context`: Data retrieved during the episode through tools.
- `step_count` & `max_steps`.
- `history_summary`: The trajectory of actions taken.

### Action Space
A restricted action space using typed payloads:
| Action | Purpose |
| ------ | ------- |
| `classify_ticket` | Assign issue category |
| `set_priority` | Set SLA urgency |
| `request_account_details` | Retrieve internal customer DB |
| `request_order_history` | Retrieve shipping DB |
| `request_return_policy` | Retrieve policy rules |
| `draft_response` | Write a natural language reply |
| `issue_refund` | (Terminal) Trigger partial/full refund |
| `escalate_to_human` | (Terminal) Route to Tier-2 support |
| `close_ticket` | (Terminal) Complete the workflow |

### Dense Reward Design
Agents receive dense mathematical shaping rewards on each step:
- Positive rewards (+0.1) for strictly correct intermediate classifications.
- Penalties (-0.05) for duplicate lookups or malformed action structures.
- Large bonuses (+0.2) for executing the correct terminal protocol.
- Terminal grade calculation (up to 1.0) applied at `close_ticket`.

## Task Descriptions

The benchmark ships with three deterministic tasks simulating real operational load:
- **Task 1: Damaged Item (Easy)** - Straightforward refund processing. Low ambiguity. Expected to succeed perfectly on modern models.
- **Task 2: Delayed Delivery (Medium)** - Mixed signals requiring specific policy lookups. Must evaluate whether cancellation or 15% courtesy refund is appropriate.
- **Task 3: Suspected Fraud (Hard)** - High-value missing item report. The agent must synthesize fraud flags and correctly decline a refund, opting for human escalation.

## Setup and Validation

### 1. Local Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open `http://localhost:7860` for the interactive dashboard.

### 2. Docker Setup
```bash
docker build -t resolveflow .
docker run -p 7860:7860 resolveflow
```

### 3. API Endpoints
- `GET /health` : Service health.
- `POST /api/reset` : Intitialize environment.
- `POST /api/step` : Perform action.
- `GET /api/state` : Fetch full state history.

## Baseline Inference Evaluation

Run the baseline evaluation using the OpenAI client:
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4-turbo"
python inference.py
```

### Example Logs Output
```
[START]
...
[STEP] {"task": "task_001_easy", "action": {"action_type": "classify_ticket", "payload": {"label": "damaged_item"}}, "reward": {"value": 0.09, "components": {"step_penalty": -0.01, "correct_classification": 0.1}}, "done": false}
...
[END]
```

### Baseline Expected Scores
- **GPT-4-Turbo**: Expected 0.85+ 
- **Mock/Random**: ~0.0 - 0.20

## Hugging Face Spaces Deployment
The included `Dockerfile` and `app.py` configuration binds correctly to `0.0.0.0:7860`. Push the repository directly to a Hugging Face Space using the Docker runtime type to host the UI.
