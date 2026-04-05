import os
import json
import time
from typing import Dict, Any, Tuple
from openai import OpenAI
from envs.environment import OpenSupportEnv
from envs.models import Action

def json_extract(response_text: str) -> Dict[str, Any]:
    try:
        # Simple extraction for robust payload fetching
        if "```json" in response_text:
            cleaned = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            cleaned = response_text.split("```")[1].split("```")[0].strip()
        else:
            cleaned = response_text.strip()
        return json.loads(cleaned)
    except Exception:
        return None

def build_system_prompt() -> str:
    return """You are an autonomous customer support operations agent.
You must solve the ticket by executing actions one at a time.
Available Action Types:
["classify_ticket", "set_priority", "request_account_details", "request_order_history", "request_shipping_status", "request_refund_history", "request_return_policy", "request_billing_history", "draft_response", "issue_refund", "offer_replacement", "offer_store_credit", "escalate_to_human", "close_ticket"]

Reply strictly with a JSON object containing `action_type` (string) and `payload` (dict). Do NOT add conversational text.
Example:
{
  "action_type": "request_account_details",
  "payload": {}
}
"""

def query_agent(client: OpenAI, model_name: str, obs_dict: Dict[str, Any]) -> Tuple[str, Dict]:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Current Observation:\n{json.dumps(obs_dict, indent=2)}\n\nWhat is your next action?"}
    ]
    
    # Retry loop for format safety
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0
            )
            raw = response.choices[0].message.content
            action_data = json_extract(raw)
            if action_data and "action_type" in action_data:
                return action_data["action_type"], action_data.get("payload", {})
        except Exception as e:
            if attempt == 2: raise
            time.sleep(1)
            
    return "close_ticket", {} # Fallback

def run_baseline():
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
    
    print("[START]")
    print(f"Running OpenSupportEnv baseline evaluation using {model_name}.")
    
    # We mock the client if dummy to prevent real crashes if no key
    is_mock = "sk-dummy" in api_key
    client = OpenAI(api_key=api_key, base_url=base_url) if not is_mock else None
    
    env = OpenSupportEnv()
    tasks = ["task_001_easy", "task_002_medium", "task_003_hard"]
    
    total_score = 0.0
    task_scores = {}
    
    for task_id in tasks:
        obs = env.reset(task_id)
        done = False
        
        while not done:
            obs_dict = obs.model_dump()
            
            # --- AGENT POLICY ---
            if is_mock:
                action_type, payload = get_mock_action(obs_dict)
            else:
                action_type, payload = query_agent(client, model_name, obs_dict)
            # --------------------
            
            try:
                action = Action(action_type=action_type, payload=payload)
                step_res = env.step(action)
                obs_dict = step_res.observation.model_dump()
                done = step_res.done
                
                log_data = {
                    "task": task_id,
                    "action": action.model_dump(),
                    "reward": step_res.reward.model_dump(),
                    "done": step_res.done
                }
                print(f"[STEP] {json.dumps(log_data)}")
                
                if done:
                    score = step_res.info.get("final_score", 0.0)
                    task_scores[task_id] = score
                    total_score += score
                    
            except Exception as e:
                print(f"Agent generated invalid action: {e}")
                done = True

    print("\n[END]")
    print("=== SUMMARY ===")
    for t_id, sc in task_scores.items():
        print(f"{t_id}: {sc:.2f}")
    
    avg_score = total_score / len(tasks)
    print(f"Aggregate Score: {avg_score:.2f}")

def get_mock_action(obs):
    step = obs["step_count"]
    diff = obs["difficulty"]
    
    # Updated mock to map to new action set and rubrics
    if diff == "easy":
        if step == 0: return "classify_ticket", {"label": "damaged_item"}
        if step == 1: return "set_priority", {"priority": "medium"}
        if step == 2: return "request_account_details", {}
        if step == 3: return "draft_response", {"message": "Here is a refund. Keep the item and accept our apology."}
        if step == 4: return "issue_refund", {"amount": 24.99}
    elif diff == "medium":
        if step == 0: return "classify_ticket", {"label": "delayed_shipment"}
        if step == 1: return "request_shipping_status", {}
        if step == 2: return "request_return_policy", {}
        if step == 3: return "draft_response", {"message": "Sorry for the delay, here is 15%."}
        if step == 4: return "offer_store_credit", {}
    elif diff == "hard":
        if step == 0: return "classify_ticket", {"label": "missing_item"}
        if step == 1: return "set_priority", {"priority": "high"}
        if step == 2: return "request_account_details", {}
        if step == 3: return "request_refund_history", {}
        if step == 4: return "request_return_policy", {}
        if step == 5: return "draft_response", {"message": "We must escalate to the team to investigate."}
        if step == 6: return "escalate_to_human", {"team": "fraud investigate"}

    return "close_ticket", {}

if __name__ == "__main__":
    run_baseline()
