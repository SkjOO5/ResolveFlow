# ResolveFlow: Customer Support Operations Benchmark

**ResolveFlow** implements the `OpenSupportEnv` environment, a realistic, multi-step customer support operations benchmark built on the **OpenEnv** specification. 

It evaluates autonomous agents strictly on information gathering, policy reasoning, safe communication, and correct operational resolutions (refunds, store credits, escalations, etc.).

## 📖 Motivation

Modern LLMs perform exceptionally well on static classification tasks but often fail in real-world operations where business logic holds hidden constraints. 
In ResolveFlow, the agent acts as a Tier-1 support agent dealing directly with customer inputs. 
It must accurately triage the ticket, correctly use internal API tools to fetch the state of the account (lifetime value, fraud flags, SLA timers), evaluate eligibility according to restricted policies, draft a compliant response, and cleanly close the case.

**This is a critical benchmark because it tests:**
- Strict policy adherence over "helpful" hallucinations.
- Efficient tool usage without duplicated calls.
- Safe escalation routing for constrained cases (like suspected fraud).
- Exact step sequence matching against a deep, mathematical Rubric.

## ⚙️ Environment Mechanics

### Observation Space
A realistic, partitioned view mimicking a Zendesk/Intercom widget:
- `customer_message`: Natural language inquiry to be parsed.
- `metadata`: Initial provided knowledge (Customer ID, Order ID).
- `revealed_context`: Keys dynamically filled as the agent queries tools (like `account`, `refund_history`, `order_status`).
- `step_count` & `max_steps`.
- `history_summary`: The trajectory of recent actions taken.
- The **hidden rubric**, score eligibility, and actual valid resolution paths are strictly hidden from the observation.

### Action Space
Agents submit typed JSON payloads choosing from:
| Action | Description |
| ------ | ------- |
| `classify_ticket` | Assign issue category |
| `set_priority` | Evaluate urgency by SLA |
| `request_account_details` | Inspect VIP/Churn risk variables |
| `request_order_history` | Look up items, amounts |
| `request_shipping_status` | Identify delivery times/zones |
| `request_refund_history` | Calculate fraud risk logic |
| `request_return_policy` | Pull instructions for decision bounds |
| `request_billing_history` | Inspect source transaction validity |
| `draft_response` | Output empathetic & compliant communication |
| `issue_refund` | (Terminal) Distribute funds |
| `offer_replacement` | (Terminal) Route physical reshipment |
| `offer_store_credit` | (Terminal) Distribute partial courtesy credits |
| `escalate_to_human` | (Terminal) Transfer to Tier-2 Fraud or Billing Ops |
| `close_ticket` | (Terminal) Conclude an already resolved flow |

### Dynamic Grader & 1.0 Normalization
The benchmark produces a `[0.0, 1.0]` score deterministically evaluating efficiency, tool accuracy, prioritization, safe policy resolution, and response quality. It heavily maps against SLA penalties, over-promising restrictions, and fraud handling.

## 🗂️ Task Difficulty Tiers

ResolveFlow ships with three deterministic task scenarios to evaluate scaling difficulty:
1. **Easy (Damaged Item)**: Short path. Agent simply checks the order and issues a policy-aligned refund natively.
2. **Medium (Delayed Delivery)**: Medium path. Relies on checking shipping SLAs and substituting an immediate cancellation request with a compliant 15% courtesy store credit instead.
3. **Hard (Suspected Fraud)**: High-value watch missing. High churn risk, multiple prior refunds flag the account. The agent must successfully navigate policy rules to explicitly decline the direct refund and pass it to human escalation gracefully.

---

## 🚀 Setup and Validation

### 1. Local UI Deployment
We provide a standalone FastAPI GUI mimicking a true operations dashboard (No React build needed):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
View the dashboard at `http://localhost:8888`.

### 2. Docker / Hugging Face Spaces
```bash
docker build -t resolveflow .
docker run -p 7860:7860 resolveflow
```
*Note: The FastAPI instance automatically listens on `0.0.0.0:*` responding properly to the standard HF Space `/health` probes.*

### 3. Baseline API Inference
Validate current SOTA models via `inference.py` conforming strictly to the OpenEnv logs format (`[START]`, `[STEP]`, `[END]`).
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4-turbo"
python inference.py
```

### Baseline Expected Scores
- **GPT-4-Turbo**: 0.82 - 0.95
- **GPT-3.5 / Models missing context capacity**: 0.45 - 0.60
- **Random Heuristics**: ~0.05
