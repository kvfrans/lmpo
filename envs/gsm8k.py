'''From https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb'''

from dataclasses import dataclass, replace
import numpy as np

from lmpo.envs.base import BaseEnv, BaseState

def extract_xml_answer(text: str) -> float:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        answer = answer.strip().replace(",", "").replace("$", "")
        return float(answer)
    except:
        return -100

def has_formatting(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        raise ValueError("Expected text to contain '####' for answer extraction.")
    return float(text.split("####")[1].strip().replace(",", "").replace("$", ""))

SYSTEM_PROMPT = """
Respond in the following format. Put a single number in the <answer> tag.
<think>
...
</think>
<answer>
...
</answer>"""

@dataclass(frozen=True)
class GSMState(BaseState):
    tokens: list
    correct_answer: float
    rendered: str = ""
    
class GSM8KEnv(BaseEnv):
    def __init__(self, tokenizer, train=True):
        super().__init__()
        self.tokens_per_action = 512
        self.force_answer_at = 50
        self.data_dict = {}
        self.tokenizer = tokenizer
        from datasets import load_dataset
        self.ds = load_dataset('openai/gsm8k', 'main')['train' if train else 'test']
        self.num_tasks = len(self.ds)

    def reset(self, idx):
        output_tokens = self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {"role": "user", "content": self.ds[idx]['question']},
            ],
            add_generation_prompt=True,
            enable_thinking=True
        )
        state = GSMState(tokens=output_tokens, correct_answer=extract_hash_answer(self.ds[idx]['answer']))
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.get_eos_token_id())
        action_msg = self.tokenizer.decode(action_tokens)
        reward = 0.0
        evaluated_answer = None
        if has_formatting(action_msg):
            reward = 0.1
            evaluated_answer = extract_xml_answer(action_msg)
            if abs(evaluated_answer - state.correct_answer) < 1e-6:
                reward = 1.0

        render_str = [
            f"{self.tokenizer.decode(state.tokens + action_tokens)}",
            f"Evaluated answer: {evaluated_answer}",
            f"Correct answer: {state.correct_answer}",
            f"Has formatting? {has_formatting(action_msg)}",
            f"Reward: {reward:.2f}",
        ]
        render_str = "\n".join(render_str)
        state = replace(state, tokens=state.tokens + action_tokens, rendered=render_str)
        return state, [], reward, True, {
            'valid_equation': reward > 0.0,
            'correct_answer': reward >= 1.0,
            'action_length': len(action_tokens),
        }