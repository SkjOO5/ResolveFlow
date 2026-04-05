from envs.models import Observation, Action, Reward, State, StepResult, TaskDefinition
from envs.tasks import get_task_by_id, EASY_TASK
from envs.rewards import RewardShaper
from envs.graders import Grader

class OpenSupportEnv:
    def __init__(self):
        self.current_task: TaskDefinition | None = None
        self._state: State | None = None
        
    def reset(self, task_id: str | None = None) -> Observation:
        if task_id:
            self.current_task = get_task_by_id(task_id)
        else:
            self.current_task = EASY_TASK
            
        self._state = State(
            task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            step_count=0,
            max_steps=self.current_task.max_steps,
            done=False,
            cumulative_reward=0.0
        )
        
        return self._get_observation()
        
    def _get_observation(self) -> Observation:
        summary = []
        for a in self._state.action_history:
            summary.append(a.action_type)
            
        return Observation(
            task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            customer_message=self.current_task.customer_message,
            metadata=self.current_task.visible_metadata,
            revealed_context=self._state.revealed_context,
            available_actions=[
                "classify_ticket", "set_priority", "request_account_details",
                "request_order_history", "request_shipping_status", "request_return_policy",
                "draft_response", "issue_refund", "offer_replacement",
                "escalate_to_human", "close_ticket"
            ],
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            history_summary=summary,
            done=self._state.done
        )
        
    def state(self) -> State:
        if not self._state:
            raise ValueError("Environment not initialized. Call reset() first.")
        return self._state
        
    def step(self, action: Action) -> StepResult:
        if self._state.done:
            raise ValueError("Episode already done. Call reset().")
            
        self._state.step_count += 1
        self._state.action_history.append(action)
        
        # Base processing & execution of valid tools
        if action.action_type == "request_account_details":
            self._state.revealed_context["account"] = self.current_task.internal_data_store.get("account")
        elif action.action_type == "request_order_history" or action.action_type == "request_shipping_status":
            self._state.revealed_context["order"] = self.current_task.internal_data_store.get("order")
        elif action.action_type == "request_return_policy":
            self._state.revealed_context["policy"] = self.current_task.internal_data_store.get("policy")
            
        # Terminal checks
        terminal_actions = ["close_ticket"]
        if action.action_type in terminal_actions:
            self._state.done = True
            
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True
            
        # Calculate Reward
        reward = RewardShaper.compute_step_reward(action, self._state, self.current_task)
        self._state.last_reward = reward
        self._state.cumulative_reward += reward.value
        
        # Calculate final grade if done
        info = {}
        if self._state.done:
            final_score, breakdown, summary = Grader.grade(self._state, self.current_task)
            self._state.final_score = final_score
            self._state.score_breakdown = breakdown
            self._state.terminal_summary = summary
            
            # Add terminal grade bonus
            terminal_bonus = final_score
            self._state.cumulative_reward += terminal_bonus
            reward.value += terminal_bonus
            reward.components["final_grader_bonus"] = terminal_bonus
            reward.reason += f" | Terminal grader bonus applied: {terminal_bonus:.2f}"
            
            info["final_score"] = final_score
            info["breakdown"] = breakdown
            info["terminal_summary"] = summary
            
        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self._state.done,
            info=info
        )
