from envs.models import State, Action, Reward, TaskDefinition

class RewardShaper:
    # Penalties
    PENALTY_INVALID_ACTION = -0.05
    PENALTY_DUPLICATE_ACTION = -0.03
    PENALTY_PREMATURE_CLOSE = -0.08
    PENALTY_PROHIBITED_CLAIM = -0.10
    PENALTY_EXCESS_STEP = -0.02
    
    # Rewards
    REWARD_CORRECT_CLASSIFICATION = 0.10
    REWARD_CORRECT_PRIORITY = 0.08
    REWARD_TOOL_LOOKUP = 0.05
    REWARD_POLICY_BEFORE_RISK = 0.05
    REWARD_COMPLIANT_RESPONSE = 0.10
    REWARD_CORRECT_RESOLUTION = 0.20
    REWARD_CORRECT_ESCALATION = 0.07

    @staticmethod
    def compute_step_reward(action: Action, state: State, task: TaskDefinition) -> Reward:
        val = 0.0
        components = {}
        reason = []

        # Time penalty
        val += RewardShaper.PENALTY_EXCESS_STEP
        components["step_penalty"] = RewardShaper.PENALTY_EXCESS_STEP

        past_action_types = [entry.action.action_type for entry in state.action_history]
        is_duplicate = False
        
        # Tools tracking
        if action.action_type.startswith("request_"):
            if action.action_type in past_action_types:
                val += RewardShaper.PENALTY_DUPLICATE_ACTION
                components["duplicate_penalty"] = RewardShaper.PENALTY_DUPLICATE_ACTION
                reason.append(f"Duplicated tool call: {action.action_type}")
                is_duplicate = True
            else:
                val += RewardShaper.REWARD_TOOL_LOOKUP
                components["tool_lookup"] = RewardShaper.REWARD_TOOL_LOOKUP
                reason.append(f"Useful data gathered via {action.action_type}")
                
            # Bonus for looking up policy before issuing resolution
            if action.action_type == "request_return_policy" and "issue_refund" not in past_action_types:
                val += RewardShaper.REWARD_POLICY_BEFORE_RISK
                components["safe_policy_check"] = RewardShaper.REWARD_POLICY_BEFORE_RISK
                reason.append("Safely checked policy prior to resolution.")

        # State updates
        if action.action_type == "classify_ticket" and state.classification_set is None:
            if action.payload.get("label") == task.hidden_rubric.correct_classification:
                val += RewardShaper.REWARD_CORRECT_CLASSIFICATION
                components["correct_classification"] = RewardShaper.REWARD_CORRECT_CLASSIFICATION
                reason.append("Correctly classified ticket")
                state.classification_set = task.hidden_rubric.correct_classification
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION
                components["invalid_classification"] = RewardShaper.PENALTY_INVALID_ACTION
                reason.append("Incorrect classification")

        elif action.action_type == "set_priority" and state.priority_set is None:
            if action.payload.get("priority") == task.hidden_rubric.correct_priority:
                val += RewardShaper.REWARD_CORRECT_PRIORITY
                components["correct_priority"] = RewardShaper.REWARD_CORRECT_PRIORITY
                reason.append("Correct priority assigned")
                state.priority_set = task.hidden_rubric.correct_priority
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION
                components["invalid_priority"] = RewardShaper.PENALTY_INVALID_ACTION
                reason.append("Incorrect priority")
        
        elif action.action_type == "draft_response" and not is_duplicate:
            response_text = action.payload.get("message", "").lower()
            hit_all = all(req in response_text for req in task.hidden_rubric.required_response_elements)
            hit_prohibited = any(proh in response_text for proh in task.hidden_rubric.prohibited_response_elements)
            
            if hit_all and not hit_prohibited:
                val += RewardShaper.REWARD_COMPLIANT_RESPONSE
                components["compliant_response"] = RewardShaper.REWARD_COMPLIANT_RESPONSE
                reason.append("Drafted flawless response")
                state.response_drafted = response_text
            elif hit_prohibited:
                val += RewardShaper.PENALTY_PROHIBITED_CLAIM
                components["prohibited_claim"] = RewardShaper.PENALTY_PROHIBITED_CLAIM
                reason.append("Response violates compliance constraints")
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION
                components["non_compliant_response"] = RewardShaper.PENALTY_INVALID_ACTION
                reason.append("Response missing key elements")

        elif action.action_type in ["issue_refund", "offer_replacement", "offer_store_credit"]:
            if action.action_type in task.hidden_rubric.valid_terminal_actions:
                val += RewardShaper.REWARD_CORRECT_RESOLUTION
                components["valid_resolution"] = RewardShaper.REWARD_CORRECT_RESOLUTION
                reason.append("Executed correct operational resolution")
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION * 2
                components["invalid_resolution"] = RewardShaper.PENALTY_INVALID_ACTION * 2
                reason.append("Invalid resolution for this policy")
                
        elif action.action_type == "escalate_to_human":
            if task.hidden_rubric.escalation_required:
                val += RewardShaper.REWARD_CORRECT_ESCALATION
                components["correct_escalation"] = RewardShaper.REWARD_CORRECT_ESCALATION
                reason.append("Correctly escalated sensitive ticket")
            else:
                val += RewardShaper.PENALTY_INVALID_ACTION
                components["unnecessary_escalation"] = RewardShaper.PENALTY_INVALID_ACTION
                reason.append("Unnecessary escalation wastes operations time")

        elif action.action_type == "close_ticket":
            # Check for premature closing
            valid_term_taken = any(entry.action.action_type in task.hidden_rubric.valid_terminal_actions for entry in state.action_history)
            if "close_ticket" not in task.hidden_rubric.valid_terminal_actions and not valid_term_taken:
                val += RewardShaper.PENALTY_PREMATURE_CLOSE
                components["premature_close"] = RewardShaper.PENALTY_PREMATURE_CLOSE
                reason.append("Premature ticket closure")

        if not reason:
            reason.append("Action registered")
            
        return Reward(value=val, components=components, reason=" | ".join(reason))
