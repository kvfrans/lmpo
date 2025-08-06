from dataclasses import dataclass, replace
import numpy as np

from lmpo.envs.base import BaseEnv, BaseState

@dataclass(frozen=True)
class PoemState(BaseState):
    tokens: list
    
class PoemLengthEnv(BaseEnv):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokens_per_action = 128
        self.tokenizer = tokenizer

    def reset(self, idx):
        imagenet_labels = open('inference/imagenet_labels.txt').read().splitlines()
        msg = f'Write three sentences about {imagenet_labels[idx % 1000]}'
        output_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            add_generation_prompt=True,
            enable_thinking=False
        )
        state = PoemState(tokens=output_tokens)
        return state, output_tokens

    def render(self, state):
        return self.tokenizer.decode(state.tokens)

    def step(self, state, action_tokens):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.get_eos_token_id())
        action_msg = self.tokenizer.decode(action_tokens)
        reward = len(action_msg)
        state = replace(state, tokens=state.tokens + action_tokens)
        return state, [], reward, True, {}
        