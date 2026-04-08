---
title: ResolveFlow
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_file: app.py
---

# 🎯 ResolveFlow — OpenSupportEnv Benchmark

**ResolveFlow** is a realistic, multi-step customer support operations benchmark built on the **OpenEnv** specification. It evaluates autonomous agents on information gathering, policy-grounded decision-making, safe communication, and correct operational resolution of real customer support tickets.

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v1.0-blue)](openenv.yaml)
[![HF Spaces Ready](https://img.shields.io/badge/HuggingFace-Spaces%20Ready-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Compatible-green)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#)

---

## 📖 Overview & Motivation

Modern LLMs perform well on isolated classification tasks, but they frequently fail in real operational workflows where:
- Business logic is hidden and policy-constrained
- Tool calls must be sequenced intelligently
- Wrong terminal actions (e.g., issuing a refund to a fraud account) cause catastrophic failures

**ResolveFlow** simulates a Tier-1 customer support agent working inside a realistic CRM-like environment with:
- Real customer tickets with natural language
- Hidden internal databases (account history, refund records, billing anomalies)
- Strict organizational policies governing legal resolution paths
- A multi-dimensional grader that rewards nuanced correct behavior

This is a critical benchmark because it tests:
- Strict policy adherence over "helpful" hallucinations
- Efficient tool usage without duplicate or wasteful calls
- Safe escalation routing for constrained fraud cases
- Correct step sequence matching against a structured hidden rubric

---

## ⚙️ Environment Description

`OpenSupportEnv` is a sequential decision-making environment where an agent processes a customer support ticket by choosing actions from a typed action space, revealing hidden context through tool calls, and resolving the case via a terminal action.

The environment is **deterministic**, **typed**, and fully **OpenEnv-compliant**, exposing:
- `reset(task_id)` → returns clean `Observation`
- `step(action)` → returns `StepResult` with `Observation`, `Reward`, `done`, `info`
- `current_state()` → returns full `State`

---

## 🧩 Observation Space

Each observation is a structured JSON object exposing:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Current task identifier |
| `difficulty` | `str` | `easy` / `medium` / `hard` |
| `customer_message` | `str` | Raw customer natural language complaint |
| `metadata` | `dict` | Initial visible context (Customer ID, Order ID) |
| `revealed_context` | `dict` | Data unlocked via tool calls (account, policy, etc.) |
| `available_actions` | `list[str]` | Valid action types in current state |
| `step_count` | `int` | Current step number |
| `max_steps` | `int` | Episode step budget |
| `history_summary` | `list[str]` | Last 3 actions taken (compact trace) |
| `done` | `bool` | Whether episode has ended |

> **Note:** The hidden rubric, grader truth, and valid resolution paths are **never** exposed in the observation.

---

## 🎮 Action Space

Agents submit typed JSON actions with `action_type` (string) and `payload` (dict):

| Action | Type | Description |
|---|---|---|
| `classify_ticket` | Info | Assign issue category (e.g. `damaged_item`) |
| `set_priority` | Info | Evaluate urgency (`low` / `medium` / `high`) |
| `request_account_details` | Tool | Reveals account tier, fraud flag, churn risk |
| `request_order_history` | Tool | Reveals order items, amount, delivery status |
| `request_shipping_status` | Tool | Reveals shipping timeline and SLA status |
| `request_refund_history` | Tool | Reveals count and amount of prior refunds |
| `request_return_policy` | Tool | Reveals applicable policy for this case |
| `request_billing_history` | Tool | Reveals payment method and chargeback flags |
| `draft_response` | Communication | Write a compliant customer-facing message |
| `issue_refund` | **Terminal** | Authorize monetary refund |
| `offer_replacement` | **Terminal** | Authorize physical item reship |
| `offer_store_credit` | **Terminal** | Issue courtesy store credit |
| `escalate_to_human` | **Terminal** | Route case to Tier-2 Fraud / Billing Ops |
| `close_ticket` | **Terminal** | Close as resolved |

---

## 📊 Reward Design

Rewards are **dense** (issued every step) and **shaped** to encourage realistic agent behavior:

| Signal | Value | Condition |
|---|---|---|
| `tool_lookup` | `+0.03` | Useful context retrieval (non-duplicate) |
| `correct_classification` | `+0.10` | Correctly classified issue type |
| `correct_priority` | `+0.05` | Correct urgency level set |
| `duplicate_lookup` | `-0.03` | Repeated tool call |
| `step_penalty` | `-0.02` | Costly step beyond minimum needed |
| `policy_violation` | `-0.20` | Attempted prohibited action (refund on fraud) |
| `terminal_bonus` | up to `+0.50` | Final score contribution at episode end |

All rewards are bounded. Cumulative reward reflects trajectory quality. A terminal grader produces the normalized final score in `[0.0, 1.0]`.

---

## 🏆 Grader Design

The deterministic grader evaluates 7 dimensions weighted into a final `[0.0, 1.0]` score:

| Dimension | Weight | Description |
|---|---|---|
| Classification | 20% | Correct issue category |
| Priority | 10% | Correct urgency level |
| Tool Usage | 20% | All required lookups completed |
| Policy Compliance | 20% | No prohibited actions taken |
| Resolution | 20% | Valid terminal action for this case |
| Response Quality | 5% | Customer message hits required semantic elements |
| Efficiency | 5% | Resolved within budget step threshold |

The grader also produces an **Episode Audit** — a human-readable list of what was correct, what was wrong, and what was missing. No LLM is used in grading.

---

## 🗂️ Task Difficulty Tiers

ResolveFlow ships with three deterministic task scenarios:

### 1. 🟢 Easy — Damaged Item Refund (`task_001_easy`)
- **Customer**: Item arrived damaged. Wants a refund.
- **Correct path**: Classify → set priority → (optional tools) → draft response → issue refund
- **Max steps**: 10
- **Key challenge**: Correct classification and completing a refund without over-promising

### 2. 🟡 Medium — Delayed Delivery (`task_002_medium`)
- **Customer**: Order hasn't arrived. Wants full cancellation.
- **Correct path**: Classify → check shipping status → check policy → draft apology with 15% credit → offer store credit
- **Max steps**: 15
- **Key challenge**: Full refund is **prohibited**. Agent must discover the policy and issue only the bounded courtesy credit

### 3. 🔴 Hard — High-Value Suspected Fraud (`task_003_hard`)
- **Customer**: Claims a $900 watch box arrived empty. Demands immediate card refund.
- **Correct path**: Classify → check account (fraud flag) → check refund history (4 prior refunds, $3200 total) → check billing (chargeback warning) → check policy → escalate to human
- **Max steps**: 20
- **Key challenge**: Fraud flag, refund abuse history, and billing anomaly all contraindicate any direct resolution. Any refund/replacement is a **policy violation** and results in a heavily penalized score

---

## 🛠️ Key Architectural Features

### Robust Action History Log
All steps, rewards, and reward components are archived into `action_history`. The UI renders per-step reward chips (e.g., `+0.05 tool_lookup`, `-0.03 duplicate`) for full trajectory transparency.

### Verified Internal Context
Agents must call tools to uncover hidden CRM data. The dashboard highlights newly retrieved context with a `✨ NEW` indicator — making tool usage feel materially meaningful.

### Terminal Audit Card
At episode completion, a human-readable audit panel renders what was correct, what was missed, and where fatal policy violations occurred — making grading decisions fully explainable.

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Node.js 20+ (for frontend build)

### Local Development

```bash
# Clone and set up Python env
git clone <your-repo-url>
cd ResolveFlow

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Serve (frontend already pre-built in /static)
python app.py
```

Open `http://localhost:7860` in your browser.

### Rebuild Frontend (optional)

```bash
cd frontend
npm install
npm run build
# Output goes to frontend/dist, copy to static/
cp -r dist/* ../static/
```

---

## 🐳 Docker Usage

```bash
docker build -t resolveflow .
docker run -p 7860:7860 resolveflow
```

After startup verify:
- `http://localhost:7860` — UI loads
- `http://localhost:7860/health` — returns `{"status": "ok"}`
- `POST http://localhost:7860/api/reset?task_id=task_001_easy` — resets environment

---

## ☁️ Hugging Face Spaces Deployment

This project is fully compatible with HF Spaces (Docker SDK):

1. Push repository to your HF Space repo
2. Set environment secrets in HF Space settings:
   - `OPENAI_API_KEY`
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
3. The app auto-binds to `0.0.0.0:7860` and responds to all health probes

---

## 🔑 Environment Variables

All required variables must be set before running inference:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI-compatible API key |
| `API_BASE_URL` | ✅ Yes | API endpoint base URL |
| `MODEL_NAME` | ✅ Yes | Model to evaluate (e.g., `gpt-4o-mini`) |
| `HF_TOKEN` | ✅ Yes | Hugging Face token |
| `PORT` | Optional | Server port (default: `7860`) |

> The application **fails immediately with a clear error** if any required variable is missing. No dummy fallbacks.

Copy `.env.example` to `.env` and fill in your values (never commit `.env` to source control).

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="sk-your-real-key"
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="hf_your_token"
python inference.py
```

### Linux / macOS bash

```bash
export OPENAI_API_KEY="sk-your-real-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="hf_your_token"
python inference.py
```

---

## 🤖 Baseline Inference

`inference.py` (root level) runs a full baseline evaluation across all 3 tasks using any OpenAI-compatible model:

```bash
python inference.py
```

Output follows strict OpenEnv log format:
- `[START]` — session opens
- `[STEP] {...}` — per-step JSON log with action, reward, done
- `[END]` — session closes with summary scores

### Baseline Expected Scores

| Model | Easy | Medium | Hard | Avg |
|---|---|---|---|---|
| GPT-4-Turbo | ~0.95 | ~0.82 | ~0.68 | ~0.82 |
| GPT-4o-mini | ~0.88 | ~0.70 | ~0.52 | ~0.70 |
| GPT-3.5 | ~0.65 | ~0.48 | ~0.30 | ~0.48 |
| Random Heuristic | ~0.10 | ~0.08 | ~0.05 | ~0.08 |

> Scores represent normalized `[0.0, 1.0]` grader outputs averaged across 3 deterministic runs.

---

## 🧪 Running Tests

```bash
# Activate venv first
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Tests cover:
- Environment reset creates valid observation
- Step logic and reward signals
- Flawless execution scores 1.0
- Invalid actions are penalized
- Policy violations drop scores below 0.5
- Action history records step, reward, and components
- Cumulative reward consistency
- Terminal state completeness (final\_score, breakdown, audit)
- Score bounds remain in `[0.0, 1.0]`
- API health endpoint

---

## 📁 Repository Structure

```
ResolveFlow/
├── app.py                  # FastAPI server (reset, step, state, health)
├── inference.py            # OpenEnv-compliant baseline runner
├── openenv.yaml            # OpenEnv spec manifest
├── Dockerfile              # Multi-stage build (Node + Python)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── envs/
│   ├── environment.py      # OpenSupportEnv core logic
│   ├── models.py           # Typed Pydantic models
│   ├── tasks.py            # Task definitions (easy/medium/hard)
│   ├── rewards.py          # Dense reward shaping logic
│   ├── graders.py          # Deterministic terminal grader
│   └── datasets.py         # Hidden CRM/account mock datasets
├── frontend/               # React + Vite premium UI
├── static/                 # Built frontend assets (served by FastAPI)
└── tests/
    └── test_env.py         # Submission validation test suite
```

---

## ⚠️ Limitations & Future Work

- **Mock datasets**: Internal CRM data is deterministic mock data. Future versions could integrate a live dataset or synthetic data generator.
- **Single-turn grader**: Response quality currently checks keyword presence. Future work could use a lightweight NLI classifier.
- **Task count**: Currently 3 tasks. Future expansion could support 10+ scenarios across verticals (e-commerce, SaaS, healthcare).
- **Multi-agent**: Currently single-agent. Future versions could evaluate supervisor + worker agent pipelines.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
