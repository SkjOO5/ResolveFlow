from envs.models import State, TaskDefinition
from envs.scoring import strict_score, safe_divide

class Grader:
    @staticmethod
    def grade(state: State, task: TaskDefinition) -> tuple[float, dict[str, float], str, list[str]]:
        score = 0.0
        audit = []
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
        if state.step_count <= (task.max_steps * 0.6):
            breakdown["efficiency"] = strict_score(1.0)
            audit.append(f"Highly efficient: Resolved in {state.step_count} steps.")
        elif state.step_count <= task.max_steps:
            efficiency_ratio = 1.0 - ((state.step_count - (task.max_steps * 0.6)) / task.max_steps)
            breakdown["efficiency"] = strict_score(max(0.0, efficiency_ratio))
            audit.append(f"Acceptable efficiency: Took {state.step_count} steps.")
        else:
            breakdown["efficiency"] = strict_score(0.1)  # Penalized but not exact 0
            audit.append(f"Inefficient: Exceeded max intended steps.")
            
        # Classification (20%)
        if state.classification_set == task.hidden_rubric.correct_classification:
            breakdown["classification"] = strict_score(1.0)
            audit.append(f"Correctly classified issue as {task.hidden_rubric.correct_classification}.")
        else:
            breakdown["classification"] = strict_score(0.1)  # Penalized but not exact 0
            audit.append(f"Incorrect classification. Expected: {task.hidden_rubric.correct_classification}. Got: {state.classification_set}.")
            
        # Priority (10%)
        if state.priority_set == task.hidden_rubric.correct_priority:
            breakdown["priority"] = strict_score(1.0)
            audit.append(f"Correctly assessed priority as {task.hidden_rubric.correct_priority}.")
        else:
            breakdown["priority"] = strict_score(0.15)  # Penalized but not exact 0
            audit.append(f"Incorrect priority. Expected: {task.hidden_rubric.correct_priority}.")
            
        # Tool usage relevance (20%)
        past_actions = [entry.action.action_type for entry in state.action_history]
        missing_tools = [t for t in task.hidden_rubric.required_tool_calls if t not in past_actions]
        if not missing_tools:
            breakdown["tool_usage"] = strict_score(1.0)
            audit.append("Successfully utilized all required internal tools.")
        else:
            pct_missing = len(missing_tools) / max(1, len(task.hidden_rubric.required_tool_calls))
            tool_score = max(0.1, 1.0 - pct_missing)
            breakdown["tool_usage"] = strict_score(tool_score)
            audit.append(f"Missed critical tool lookups: {', '.join(missing_tools)}. Context retrieval insufficient.")
            
        # Policy Compliance (20%) - Checking if agent broke refund rules
        has_policy_breach = False
        if "issue_refund" in past_actions and not task.hidden_rubric.refund_eligible:
            has_policy_breach = True
            audit.append("Fatal policy violation: Issued refund when customer was ineligible.")
        if "offer_replacement" in past_actions and not task.hidden_rubric.replacement_eligible:
            has_policy_breach = True
            audit.append("Policy violation: Offered replacement when ineligible.")
        if "escalate_to_human" not in past_actions and task.hidden_rubric.escalation_required:
            has_policy_breach = True
            audit.append("Fatal error: Failed to escalate a mandatory human-review case.")
            
        if not has_policy_breach:
            breakdown["policy_compliance"] = strict_score(1.0)
        else:
            breakdown["policy_compliance"] = strict_score(0.05)  # Penalized but not exact 0
            
        # Resolution Correctness (20%)
        terminals_taken = [a for a in past_actions if a in ["issue_refund", "offer_replacement", "offer_store_credit", "escalate_to_human", "close_ticket"]]
        # Must have taken at least one valid terminal action
        is_valid_term = any(t in task.hidden_rubric.valid_terminal_actions for t in terminals_taken)
        if is_valid_term and not has_policy_breach:
            breakdown["resolution"] = strict_score(1.0)
            audit.append(f"Resolution path was correct and compliant.")
        elif not is_valid_term:
            breakdown["resolution"] = strict_score(0.2)  # Penalized but not exact 0
            audit.append(f"Suboptimal resolution. Expected one of: {', '.join(task.hidden_rubric.valid_terminal_actions)}.")
        else:
            breakdown["resolution"] = strict_score(0.1)  # Policy violation affects resolution

        # Response Quality (5%)
        if state.response_drafted:
            draft = state.response_drafted.lower()
            hit_all = all(req in draft for req in task.hidden_rubric.required_response_elements)
            hit_prohibited = any(proh in draft for proh in task.hidden_rubric.prohibited_response_elements)
            if hit_all and not hit_prohibited:
                breakdown["response_quality"] = strict_score(1.0)
                audit.append("Customer response met all semantic criteria.")
            elif not hit_prohibited:
                breakdown["response_quality"] = strict_score(0.5) 
                audit.append("Customer response missed key required topics.")
            else:
                breakdown["response_quality"] = strict_score(0.05)  # Prohibited word present, penalized but not exact 0
                audit.append("Customer response contained strictly prohibited commitments.")
        elif not task.hidden_rubric.required_response_elements:
            breakdown["response_quality"] = strict_score(1.0)  # not strictly required
            audit.append("Customer communication was adequate.")
        else:
            breakdown["response_quality"] = strict_score(0.1)  # No response drafted despite being implicitly required
            audit.append("No response drafted despite being implicitly required.")
            
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
        
        # Apply strict scoring to the final result
        final_score = strict_score(final_score)
        
        summary = (
            f"Classification: {breakdown['classification']*100:.0f}% | "
            f"Priority: {breakdown['priority']*100:.0f}% | "
            f"Tools: {breakdown['tool_usage']*100:.0f}% | "
            f"Policy: {breakdown['policy_compliance']*100:.0f}% | "
            f"Resolution: {breakdown['resolution']*100:.0f}% | "
            f"Response: {breakdown['response_quality']*100:.0f}%"
        )
        
        return final_score, breakdown, summary, audit
