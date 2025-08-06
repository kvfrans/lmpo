'''From https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb and https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/games/countdown.py'''

from dataclasses import dataclass, replace
import numpy as np
import re
import sympy
from sympy import Symbol, symbols
from sympy.parsing.sympy_parser import parse_expr

from lmpo.envs.base import BaseEnv, BaseState

SYSTEM_PROMPT = "You are a helpful assistant. You first think about the reasoning process in the mind, and then provide the user with the answer."
USER_PROMPT = "Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. Think for only ten sentences, then return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."

@dataclass(frozen=True)
class CountdownState(BaseState):
    tokens: list
    numbers: list
    correct_answer: int
    rendered: str = ""

def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer
    except:
        return "Parsing error"

def valid_equation(state: CountdownState, text: str) -> bool:
    try:
        equation_str = extract_xml_answer(text)
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(state.numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False
    
def evaluate_equation(equation_str) -> int | None:
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None
    
def generate_candidate_expression(num_terms, rng):
    numbers = [rng.randint(100) for _ in range(num_terms)]
    syms = symbols(f"x:{num_terms}")
    expr = syms[0]

    for i in range(1, num_terms):
        op = np.random.choice(["+", "-", "*", "/"])
        if op == "+":
            expr = expr + syms[i]
        elif op == "-":
            expr = expr - syms[i]
        elif op == "*":
            expr = expr * syms[i]
        else:  # division
            # Handle division carefully to ensure integer results
            if numbers[i] != 0:  # Avoid division by zero
                # Get current value after substituting previous numbers
                current = int(expr.subs({sym: num for sym, num in zip(syms[:i], numbers[:i])}))
                # Try each remaining number to find one that divides evenly
                remaining = [n for n in numbers[i:] if n != 0]
                # rng.shuffle(remaining)  # Randomize order for variety
                np.random.shuffle(remaining)
                found_divisor = False
                for div in remaining:
                    if current % div == 0:  # Check if divides evenly
                        numbers[i] = div
                        expr = expr / syms[i]
                        found_divisor = True
                        break
                if not found_divisor:
                    # If no number divides evenly, fallback to subtraction
                    expr = expr - syms[i]
            else:
                # Fallback to addition for zero
                expr = expr + syms[i]

    return expr, numbers, syms

def generate_expression(rng):
    num_terms = 4

    max_attempts = 100
    for attempt in range(max_attempts):
        try:
            expr, numbers, syms = generate_candidate_expression(num_terms, rng)

            # Substitute actual numbers to get target
            subs = {sym: num for sym, num in zip(syms, numbers)}
            target = int(expr.subs(subs))

            # Convert to string expression
            expr_str = str(expr)
            for i, sym in enumerate(syms):
                expr_str = expr_str.replace(str(sym), str(numbers[i]))

            # Ensure target is within bounds
            if target > 0 and target <= 1000:
                return expr_str, numbers, target

        except (ValueError, ZeroDivisionError):
            continue

    raise ValueError(f"Failed to generate valid expression after {max_attempts} attempts")
    
class CountdownEnv(BaseEnv):
    def __init__(self, tokenizer):
        super().__init__() 
        self.tokens_per_action = 512
        self.force_answer_at = 50
        self.data_dict = {}
        self.tokenizer = tokenizer

    def reset(self, idx):
        rng = np.random.RandomState(idx)
        _, numbers, target = generate_expression(rng)
        rng.shuffle(numbers)
        output_tokens = self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(target=target, numbers=numbers)},
            ],
            add_generation_prompt=True,
            enable_thinking=True
        )
        state = CountdownState(tokens=output_tokens, numbers=numbers, correct_answer=target)
        return state, output_tokens

    def render(self, state):
        return state.rendered

    def step(self, state, action_tokens):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.get_eos_token_id())
        action_msg = self.tokenizer.decode(action_tokens)
        equation = extract_xml_answer(action_msg)
        reward = 0.0
        evaluated_answer = None
        if valid_equation(state, action_msg):
            reward = 0.1
            evaluated_answer = evaluate_equation(equation)
            if evaluated_answer is not None:
                if abs(evaluated_answer - state.correct_answer) < 1e-6:
                    reward = 1.0

        render_str = [
            f"{self.tokenizer.decode(state.tokens + action_tokens)}",
            f"Evaluated answer: {evaluated_answer}",
            f"Correct answer: {state.correct_answer}",
            f"Valid equation? {valid_equation(state, action_msg)}",
            f"Reward: {reward:.2f}",
        ]
        render_str = "\n".join(render_str)
        state = replace(state, tokens=state.tokens + action_tokens, rendered=render_str)
        return state, [], reward, True, {
            'valid_equation': reward > 0.0,
            'correct_answer': reward >= 1.0,
            'action_length': len(action_tokens),
        }