'''From https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb'''

from dataclasses import dataclass, replace
import numpy as np

from lmpo.envs.base import BaseEnv, BaseState
from lmpo.envs.math_utils import grade_answer

def extract_xml_answer(text: str) -> float:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        answer = answer.strip().replace(",", "").replace("$", "")
        return answer
    except:
        return -100

def has_formatting(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        raise ValueError("Expected text to contain '####' for answer extraction.")
    return float(text.split("####")[1].strip().replace(",", "").replace("$", ""))

SYSTEM_PROMPT = """
Respond in the following format. Put the final answer within the <answer> tag, use latex \\frac{a}{b} for fractions.
<think>
...
</think>
<answer>
...
</answer>"""

@dataclass(frozen=True)
class MathEnvState(BaseState):
    tokens: list
    correct_answer: float
    rendered: str = ""
    reward: float = -1.0
    
class MathEnv(BaseEnv):
    def __init__(self, tokenizer, env_name):
        super().__init__()
        self.tokens_per_action = 512
        self.force_answer_at = 50
        self.data_dict = {}
        self.tokenizer = tokenizer
        self.env_name = env_name
        from datasets import load_dataset
        if env_name == 'gsm8k_train':
            ds = load_dataset('openai/gsm8k', 'main')['train']
        elif env_name == 'gsm8k_test':
            ds = load_dataset('openai/gsm8k', 'main')['test']
        elif env_name == 'deepscaler':
            ds = load_dataset('agentica-org/DeepScaleR-Preview-Dataset')['train']
        elif env_name == 'dapo':
            raise ValueError("TODO: Needs work. Why does the HF dataset have 1.8M samples?")
            ds = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k')['train']
        elif env_name == 'hicra':
            # DAPO + Deepscaler, apparently.
            ds = load_dataset('JasperHaozhe/HICRA_RLDATA_Math')['train']
        elif env_name == 'e3-easy':
            ds = load_dataset("CMU-AIRe/e3-math-easy", split="train")
        elif env_name == 'e3-medhard':
            ds = load_dataset("CMU-AIRe/e3-math-medhard", split="train")
        elif env_name == 'aime2024':
            ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        elif env_name == 'aime2025':
            ds = load_dataset("MathArena/aime_2025", split="train")
        elif env_name == 'math500':
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        elif env_name == 'hmmt2025':
            ds = load_dataset("MathArena/hmmt_feb_2025", split="train")
        elif env_name == 'amc2023':
            ds = load_dataset("zwhe99/amc23", split="test")
        else:
            raise ValueError(f"Unknown math-environment name: {env_name}")
        self.ds = ds.shuffle(seed=42)
        self.num_tasks = len(self.ds)

    def get_question(self, ds_dict):
        if 'question' in ds_dict:
            return ds_dict['question']
        elif 'problem' in ds_dict:
            return ds_dict['problem']
        elif 'prompt' in ds_dict:
            question = ds_dict['prompt'][0]['content']
            question = question.split("Let's think")[0]
            question = question.split("Please reason")[0]
            return question
        raise ValueError("No known prompt field found.")
    
    def get_answer(self, ds_dict):
        if 'reward_model' in ds_dict:
            return ds_dict['reward_model']['ground_truth']
        elif 'answer' in ds_dict:
            if 'gsm8k' in self.env_name:
                return extract_hash_answer(ds_dict['answer'])
            return ds_dict['answer']
        elif 'solution' in ds_dict:
            return ds_dict['solution']
        raise ValueError("No known answer field found.")


    def reset(self, idx):
        question = self.get_question(self.ds[idx])
        output_tokens = self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            add_generation_prompt=True,
            enable_thinking=True
        )
        answer = self.get_answer(self.ds[idx])
        state = MathEnvState(tokens=output_tokens, correct_answer=answer, task_id=idx)
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens):
        action_tokens_clean = self.clean_action(action_tokens, self.tokenizer.get_eos_token_id())
        action_msg = self.tokenizer.decode(action_tokens_clean)

        is_answer_forced = (len(action_tokens) - action_tokens.index(9217) == self.force_answer_at - 3)
        reasoning_length = len(self.tokenizer.encode(action_msg.split("<answer>")[0]))
        answer_length = 0

        planning_words = ["But", "Wait", "So,", "Alternatively", "However", "Maybe,", "Hmm", "Okay", "Now,", "Then,"]
        planning_words = [w.lower() for w in planning_words]
        action_msg_lower = action_msg.lower()
        planning_word_count = sum(action_msg_lower.count(w) for w in planning_words)

        reward = 0.0
        evaluated_answer = None
        if has_formatting(action_msg):
            reward = 0.1
            evaluated_answer = extract_xml_answer(action_msg)
            try:
                if abs(float(evaluated_answer) - float(state.correct_answer)) < 1e-6:
                    reward = 1.0
            except:
                pass
            if grade_answer(evaluated_answer, state.correct_answer):
                reward = 1.0
            answer_length = len(self.tokenizer.encode(evaluated_answer))

        render_str = [
            f"{self.tokenizer.decode(state.tokens + action_tokens_clean)}",
            f"Evaluated answer: {evaluated_answer}",
            f"Correct answer: {state.correct_answer}",
            f"Has formatting? {has_formatting(action_msg)}",
            f"Reward: {reward:.2f}",
        ]
        render_str = "\n".join(render_str)
        state = replace(state, tokens=state.tokens + action_tokens_clean, rendered=render_str, reward=reward)
        return state, [], reward, True, {
            'valid_equation': reward > 0.0,
            'correct_answer': reward >= 1.0,
            'action_length': len(action_tokens_clean),
            'reasoning_length': reasoning_length,
            'answer_length': answer_length,
            'planning_word_count': planning_word_count,
            'is_answer_forced': is_answer_forced,
        }