from dataclasses import dataclass

@dataclass(frozen=True)
class BaseState:
    task_id: int

    def render(self) -> str:
        raise NotImplementedError

class BaseEnv():
    """Basic class for an LLM RL environment."""
    tokens_per_action = 32
    force_answer_at = -1
    num_tasks = -1  # -1 means no limit, otherwise it is the number of tasks in the dataset.

    def __init__(self):
        pass

    def reset(self, idx): # return a state, and an initial output_tokens.
        raise NotImplementedError

    def step(self, state, action_tokens):
        # Returns: (state, output_tokens, reward, is_done, infos)
        raise NotImplementedError

    def clean_action(self, action_tokens, end_token):
        try:
            index = action_tokens.index(end_token)
            return action_tokens[:index + 1]
        except ValueError:
            return action_tokens
    
    def step_list(self, states, action_tokens):
        assert len(states) == len(action_tokens)
        new_states = []
        new_output_tokens = []
        new_rewards = []
        new_is_dones = []
        new_infos = {}
        for state, ac in zip(states, action_tokens):
            new_state, output_tokens, reward, is_done, infos = self.step(state, ac)
            new_states.append(new_state)
            new_output_tokens.append(output_tokens)
            new_rewards.append(reward)
            new_is_dones.append(is_done)
            for k, v in infos.items():
                if k not in new_infos:
                    new_infos[k] = []
                new_infos[k].append(v) 
        return new_states, new_output_tokens, new_rewards, new_is_dones, new_infos