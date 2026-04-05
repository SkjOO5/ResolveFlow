from envs.models import State, TaskDefinition

class Grader:
    @staticmethod
    def grade(state: State, task: TaskDefinition) -> tuple[float, dict[str, float], str]:
        """
        Returns score [0.0, 1.0], breakdown dict, and summary string
        """
        score = 0.0
        breakdown = {
            "classification": 0.0,
            "priority": 0.0,
            "tool_usage": 0.0,
            "resolution": 0.0,
            "response": 0.0,
            "efficiency": 0.0
        }
        
        # Max steps penalty / Efficiency
        if state.step_count <= task.max_steps * 0.8:
            breakdown["efficiency"] = 1.0
        elif state.step_count <= task.max_steps:
            breakdown["efficiency"] = 0.5
            
        # Classification (20%)
        if state.classification_set == task.hidden_rubric.correct_classification:
            breakdown["classification"] = 1.0
            
        # Priority (15%)
        if state.priority_set == task.hidden_rubric.correct_priority:
            breakdown["priority"] = 1.0
            
        # Tool usage (20%)
        past_actions = [a.action_type for a in state.action_history]
        missing_tools = [t for t in task.hidden_rubric.required_tool_calls if t not in past_actions]
        if not missing_tools:
            breakdown["tool_usage"] = 1.0
        else:
            partial_tools = 1.0 - (len(missing_tools) / max(1, len(task.hidden_rubric.required_tool_calls)))
            breakdown["tool_usage"] = partial_tools
            
        # Resolution (25%)
        resolution_taken = [a for a in past_actions if a in ["issue_refund", "offer_replacement", "escalate_to_human", "close_ticket"]]
        is_valid_term = any(t in task.hidden_rubric.valid_terminal_actions for t in resolution_taken)
        
        # specific hard task check
        if task.difficulty == "hard":
            if "issue_refund" in resolution_taken:
                # catastrophic failure for hard task representing fraud
                breakdown["resolution"] = 0.0
            elif "escalate_to_human" in resolution_taken:
                breakdown["resolution"] = 1.0
        else:
            if is_valid_term:
                breakdown["resolution"] = 1.0
                
        # Response compliance (15%)
        if state.response_drafted:
            draft = state.response_drafted.lower()
            hit_all = all(req in draft for req in task.hidden_rubric.required_response_elements)
            hit_prohibited = any(proh in draft for proh in task.hidden_rubric.prohibited_response_elements)
            if hit_all and not hit_prohibited:
                breakdown["response"] = 1.0
            elif not hit_prohibited:
                breakdown["response"] = 0.5 # Missing elements but didn't say anything prohibited
        elif not task.hidden_rubric.required_response_elements:
            breakdown["response"] = 1.0 # not required
            
        # Weighted sum:
        final_score = (
            0.20 * breakdown["classification"] +
            0.15 * breakdown["priority"] +
            0.20 * breakdown["tool_usage"] +
            0.25 * breakdown["resolution"] +
            0.15 * breakdown["response"] +
            0.05 * breakdown["efficiency"]
        )
        
        final_score = max(0.0, min(1.0, final_score))
        
        summary = (
            f"Classification: {breakdown['classification']*100}% | "
            f"Priority: {breakdown['priority']*100}% | "
            f"Tools: {breakdown['tool_usage']*100}% | "
            f"Resolution: {breakdown['resolution']*100}% | "
            f"Response: {breakdown['response']*100}%"
        )
        
        return final_score, breakdown, summary
