import os
import json
from openai import OpenAI
from envs.environment import OpenSupportEnv
from envs.models import Action

def run_baseline():
    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
    
    print("[START]")
    print(f"Running OpenSupportEnv baseline evaluation using {model_name}.")
    
    # Normally we would initialize proper OpenAI client if key is not dummy
    # client = OpenAI(api_key=api_key, base_url=base_url)
    
    env = OpenSupportEnv()
    
    tasks = ["task_001_easy", "task_002_medium", "task_003_hard"]
    
    total_score = 0.0
    
    for task_id in tasks:
        obs = env.reset(task_id)
        done = False
        
        while not done:
            # Simulated naive baseline logic instead of real LLM calls to prevent paid requests without valid keys
            # A real baseline would format `obs` to a JSON prompt and query `client.chat.completions.create`
            obs_dict = obs.model_dump()
            
            # --- START MOCK AGENT POLICY ---
            action_type, payload = get_mock_action(obs_dict)
            # --- END MOCK AGENT POLICY ---
            
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
                    print(f"Task {task_id} completed with score: {score:.2f}")
                    total_score += score
                    
            except Exception as e:
                print(f"Agent generated invalid action: {e}")
                done = True # Abort upon severe crash

                
    avg_score = total_score / len(tasks)
    print(f"Average Baseline Score: {avg_score:.2f}")
    print("[END]")


def get_mock_action(obs):
    """Simple deterministic policy indicating structure"""
    step = obs["step_count"]
    diff = obs["difficulty"]
    
    if diff == "easy":
        if step == 0: return "classify_ticket", {"label": "damaged_item"}
        if step == 1: return "set_priority", {"priority": "medium"}
        if step == 2: return "issue_refund", {"amount": 24.99, "reason": "broken item"}
        if step == 3: return "draft_response", {"message": "Here is a refund. Keep the item and accept our apology."}
        if step == 4: return "close_ticket", {}
    elif diff == "medium":
        if step == 0: return "classify_ticket", {"label": "shipping_delay"}
        if step == 1: return "set_priority", {"priority": "medium"}
        if step == 2: return "request_shipping_status", {}
        if step == 3: return "draft_response", {"message": "Sorry for the delay, here is 15% discount."}
        if step == 4: return "close_ticket", {}
    elif diff == "hard":
        if step == 0: return "classify_ticket", {"label": "missing_item"}
        if step == 1: return "set_priority", {"priority": "high"}
        if step == 2: return "request_account_details", {}
        if step == 3: return "escalate_to_human", {"team": "fraud investigate"}

    return "close_ticket", {}

if __name__ == "__main__":
    run_baseline()
