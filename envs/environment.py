import json
from envs.models import Observation, Action, StepResult, State, ActionType
from envs.tasks import get_task_by_id
from envs.rewards import RewardShaper
from envs.graders import Grader

class OpenSupportEnv:
    def __init__(self):
        self.state: State = None
        self._task = None
        
        self.available_actions_list = [
            "classify_ticket", "set_priority", "request_account_details",
            "request_order_history", "request_shipping_status", "request_refund_history",
            "request_return_policy", "request_billing_history", "draft_response",
            "issue_refund", "offer_replacement", "offer_store_credit", "escalate_to_human", "close_ticket"
        ]

    def reset(self, task_id: str) -> Observation:
        self._task = get_task_by_id(task_id)
        
        self.state = State(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            step_count=0,
            max_steps=self._task.max_steps,
            done=False,
            cumulative_reward=0.0,
            action_history=[],
            revealed_context={},
            available_actions=self.available_actions_list
        )
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self.state.done:
            return StepResult(
                observation=self._make_observation(),
                reward=self.state.last_reward,
                done=True,
                info={"error": "Episode already done."}
            )

        self.state.step_count += 1
        reward = RewardShaper.compute_step_reward(action, self.state, self._task)
        self.state.cumulative_reward += reward.value
        self.state.last_reward = reward
        
        # Determine valid tool lookup keys
        tool_mapping = {
            "request_account_details": "account",
            "request_order_history": "order",
            "request_shipping_status": "order",
            "request_refund_history": "refund_history",
            "request_billing_history": "billing",
            "request_return_policy": "policy"
        }
        
        if action.action_type in tool_mapping:
            key = tool_mapping[action.action_type]
            if key in self._task.internal_data_store:
                self.state.revealed_context[key] = self._task.internal_data_store[key]

        self.state.action_history.append(action)

        terminal_actions = ["issue_refund", "offer_replacement", "offer_store_credit", "escalate_to_human", "close_ticket"]
        
        if action.action_type in terminal_actions or self.state.step_count >= self.state.max_steps:
            self.state.done = True
            final_score, breakdown, summary = Grader.grade(self.state, self._task)
            self.state.final_score = final_score
            self.state.score_breakdown = breakdown
            self.state.terminal_summary = summary
            
            # terminal bonus to align reward surface
            reward.value += (final_score * 0.5) 
            self.state.cumulative_reward += (final_score * 0.5)

        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            done=self.state.done,
            info={"final_score": self.state.final_score} if self.state.done else {}
        )

    def _make_observation(self) -> Observation:
        history_summary = []
        for a in self.state.action_history[-3:]: # Only visible past 3 actions to force context mgmt
            payload_str = json.dumps(a.payload) 
            history_summary.append(f"{a.action_type}: {payload_str}")
            
        return Observation(
            task_id=self.state.task_id,
            difficulty=self.state.difficulty,
            customer_message=self._task.customer_message,
            metadata=self._task.visible_metadata,
            revealed_context=self.state.revealed_context,
            available_actions=self.state.available_actions,
            step_count=self.state.step_count,
            max_steps=self.state.max_steps,
            history_summary=history_summary,
            done=self.state.done
        )

    def current_state(self) -> State:
        return self.state
