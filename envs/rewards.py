from envs.models import State, Action, Reward, TaskDefinition

class RewardShaper:
    PENALTY_INVALID_ACTION = -0.05
    PENALTY_DUPLICATE_ACTION = -0.03
    PENALTY_PREMATURE_CLOSE = -0.10
    PENALTY_STEP = -0.01

    REWARD_CORRECT_CLASSIFICATION = 0.10
    REWARD_CORRECT_PRIORITY = 0.08
    REWARD_TOOL_LOOKUP = 0.05
    REWARD_COMPLIANT_RESPONSE = 0.10
    REWARD_CORRECT_RESOLUTION = 0.20

    @staticmethod
    def compute_step_reward(action: Action, state: State, task: TaskDefinition) -> Reward:
        val = 0.0
        components = {}
        reason = []

        # Time penalty
        val += RewardShaper.PENALTY_STEP
        components["step_penalty"] = RewardShaper.PENALTY_STEP
        reason.append("Standard step time penalty.")

        # Check duplicated actions
        past_action_types = [a.action_type for a in state.action_history]
        is_duplicate = False
        
        # Tools are usually called once
        if action.action_type.startswith("request_"):
            if action.action_type in past_action_types:
                val += RewardShaper.PENALTY_DUPLICATE_ACTION
                components["duplicate_penalty"] = RewardShaper.PENALTY_DUPLICATE_ACTION
                reason.append(f"Duplicated {action.action_type}.")
                is_duplicate = True
            else:
                val += RewardShaper.REWARD_TOOL_LOOKUP
                components["tool_lookup"] = RewardShaper.REWARD_TOOL_LOOKUP
                reason.append(f"Useful information gathered via {action.action_type}.")

        if action.action_type == "classify_ticket" and state.classification_set is None:
            if action.payload.get("label") == task.hidden_rubric.correct_classification:
                val += RewardShaper.REWARD_CORRECT_CLASSIFICATION
                components["correct_classification"] = RewardShaper.REWARD_CORRECT_CLASSIFICATION
                reason.append("Correctly classified ticket.")
                state.classification_set = task.hidden_rubric.correct_classification
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION
                components["invalid_classification"] = RewardShaper.PENALTY_INVALID_ACTION
                reason.append("Incorrect classification applied.")

        elif action.action_type == "set_priority" and state.priority_set is None:
            if action.payload.get("priority") == task.hidden_rubric.correct_priority:
                val += RewardShaper.REWARD_CORRECT_PRIORITY
                components["correct_priority"] = RewardShaper.REWARD_CORRECT_PRIORITY
                reason.append("Correct priority assigned.")
                state.priority_set = task.hidden_rubric.correct_priority
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION
                components["invalid_priority"] = RewardShaper.PENALTY_INVALID_ACTION
                reason.append("Incorrect priority applied.")
        
        elif action.action_type == "draft_response" and not is_duplicate:
            response_text = action.payload.get("message", "").lower()
            hit_all = all(req in response_text for req in task.hidden_rubric.required_response_elements)
            hit_prohibited = any(proh in response_text for proh in task.hidden_rubric.prohibited_response_elements)
            
            if hit_all and not hit_prohibited:
                val += RewardShaper.REWARD_COMPLIANT_RESPONSE
                components["compliant_response"] = RewardShaper.REWARD_COMPLIANT_RESPONSE
                reason.append("Drafted response compliant with policy.")
                state.response_drafted = response_text
            else:
                # Soft penalty for non compliant draft
                val += RewardShaper.PENALTY_INVALID_ACTION
                components["non_compliant_response"] = RewardShaper.PENALTY_INVALID_ACTION
                reason.append("Drafted response violates policy or is missing elements.")

        elif action.action_type in ["issue_refund", "offer_replacement", "escalate_to_human"]:
            if action.action_type in task.hidden_rubric.valid_terminal_actions:
                val += RewardShaper.REWARD_CORRECT_RESOLUTION
                components["valid_resolution"] = RewardShaper.REWARD_CORRECT_RESOLUTION
                reason.append("Executed correct resolution protocol.")
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION * 2 # heavy penalty
                components["invalid_resolution"] = RewardShaper.PENALTY_INVALID_ACTION * 2
                reason.append("Attempted invalid resolution protocol.")

        elif action.action_type == "close_ticket":
            # If closed ticket before terminal actions were taken, it's premature
            valid_term_taken = any(a.action_type in task.hidden_rubric.valid_terminal_actions for a in state.action_history)
            
            # exception: when close_ticket is ITSELF the only terminal action needed (e.g. Easy task just needs refund and close)
            # Actually, we should check if they did necessary stuff first
            if "close_ticket" in task.hidden_rubric.valid_terminal_actions and valid_term_taken:
                pass # it's fine
            elif not valid_term_taken and "draft_response" not in past_action_types:
                val += RewardShaper.PENALTY_PREMATURE_CLOSE
                components["premature_close"] = RewardShaper.PENALTY_PREMATURE_CLOSE
                reason.append("Premature ticket closure.")

        reason_str = " | ".join(reason) if reason else "Action registered."
        return Reward(value=val, components=components, reason=reason_str)
