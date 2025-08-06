'''From https://github.com/agentica-project/rllm and https://github.com/agentica-project/rllm/blob/main/rllm/rewards/math_utils/utils.py'''

from dataclasses import dataclass, replace

from lmpo.envs.base import BaseEnv, BaseState
from lmpo.envs.deepscaler_utils import grade_answer

def extract_xml_answer(text: str) -> float:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        answer = answer.strip().replace(",", "").replace("$", "")
        return answer
    except:
        return ''

def has_formatting(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text

SYSTEM_PROMPT = """
Respond in the following format. Put the final answer within the <answer> tag, use latex \\frac{a}{b} for fractions.
<think>
...
</think>
<answer>
...
</answer>"""

@dataclass(frozen=True)
class DeepscalerState(BaseState):
    tokens: list
    correct_answer: str
    rendered: str = ""
    
class DeepscalerEnv(BaseEnv):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokens_per_action = 512
        self.force_answer_at = 50
        self.data_dict = {}
        self.tokenizer = tokenizer
        from datasets import load_dataset
        self.ds = load_dataset('agentica-org/DeepScaleR-Preview-Dataset')['train']
        self.num_tasks = len(self.ds)

    def reset(self, idx):
        output_tokens = self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {"role": "user", "content": self.ds[idx]['problem']},
            ],
            add_generation_prompt=True,
            enable_thinking=True
        )
        state = DeepscalerState(tokens=output_tokens, correct_answer=self.ds[idx]['answer'])
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
            if grade_answer(evaluated_answer, state.correct_answer):
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