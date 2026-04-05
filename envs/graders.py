from envs.models import State, TaskDefinition

class Grader:
    @staticmethod
    def grade(state: State, task: TaskDefinition) -> tuple[float, dict[str, float], str]:
        score = 0.0
        breakdown = {
            "classification": 0.0,
            "priority": 0.0,
            "tool_usage": 0.0,
            "policy_compliance": 0.0,
            "resolution": 0.0,
            "response_quality": 0.0,
            "efficiency": 0.0
        }
        
        # Efficiency (5%)
        # Stricter efficiency rule based on max_steps
        if state.step_count <= (task.max_steps * 0.6):
            breakdown["efficiency"] = 1.0
        elif state.step_count <= task.max_steps:
            # 10% penalty per step over 60%
            efficiency_ratio = 1.0 - ((state.step_count - (task.max_steps * 0.6)) / task.max_steps)
            breakdown["efficiency"] = max(0.0, efficiency_ratio)
            
        # Classification (20%)
        if state.classification_set == task.hidden_rubric.correct_classification:
            breakdown["classification"] = 1.0
            
        # Priority (10%)
        if state.priority_set == task.hidden_rubric.correct_priority:
            breakdown["priority"] = 1.0
            
        # Tool usage relevance (20%)
        past_actions = [a.action_type for a in state.action_history]
        missing_tools = [t for t in task.hidden_rubric.required_tool_calls if t not in past_actions]
        if not missing_tools:
            breakdown["tool_usage"] = 1.0
        else:
            pct_missing = len(missing_tools) / max(1, len(task.hidden_rubric.required_tool_calls))
            breakdown["tool_usage"] = max(0.0, 1.0 - pct_missing)
            
        # Policy Compliance (20%) - Checking if agent broke refund rules
        has_policy_breach = False
        if "issue_refund" in past_actions and not task.hidden_rubric.refund_eligible:
            has_policy_breach = True
        if "offer_replacement" in past_actions and not task.hidden_rubric.replacement_eligible:
            has_policy_breach = True
        if "escalate_to_human" not in past_actions and task.hidden_rubric.escalation_required:
            has_policy_breach = True
            
        if not has_policy_breach:
            breakdown["policy_compliance"] = 1.0
        else:
            breakdown["policy_compliance"] = 0.0
            
        # Resolution Correctness (20%)
        terminals_taken = [a for a in past_actions if a in ["issue_refund", "offer_replacement", "offer_store_credit", "escalate_to_human", "close_ticket"]]
        # Must have taken at least one valid terminal action
        is_valid_term = any(t in task.hidden_rubric.valid_terminal_actions for t in terminals_taken)
        if is_valid_term and not has_policy_breach:
            breakdown["resolution"] = 1.0

        # Response Quality (5%)
        if state.response_drafted:
            draft = state.response_drafted.lower()
            hit_all = all(req in draft for req in task.hidden_rubric.required_response_elements)
            hit_prohibited = any(proh in draft for proh in task.hidden_rubric.prohibited_response_elements)
            if hit_all and not hit_prohibited:
                breakdown["response_quality"] = 1.0
            elif not hit_prohibited:
                breakdown["response_quality"] = 0.5 
            else:
                breakdown["response_quality"] = 0.0 # Prohibited word present
        elif not task.hidden_rubric.required_response_elements:
            breakdown["response_quality"] = 1.0 # not strictly required
            
        # Normalized weighted sum:
        final_score = (
            0.20 * breakdown["classification"] +
            0.10 * breakdown["priority"] +
            0.20 * breakdown["tool_usage"] +
            0.20 * breakdown["policy_compliance"] +
            0.20 * breakdown["resolution"] +
            0.05 * breakdown["response_quality"] +
            0.05 * breakdown["efficiency"]
        )
        
        final_score = max(0.0, min(1.0, final_score))
        
        summary = (
            f"Classification: {breakdown['classification']*100:.0f}% | "
            f"Priority: {breakdown['priority']*100:.0f}% | "
            f"Tools: {breakdown['tool_usage']*100:.0f}% | "
            f"Policy: {breakdown['policy_compliance']*100:.0f}% | "
            f"Resolution: {breakdown['resolution']*100:.0f}% | "
            f"Response: {breakdown['response_quality']*100:.0f}%"
        )
        
        return final_score, breakdown, summary
