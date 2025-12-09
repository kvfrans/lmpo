import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast, AddedToken

############################
# This is mostly a sanity checking file, it just wraps PreTrainedTokenizerFast, but I want to know what it is doing.
#############################

class Tokenizer():
    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer

    def apply_chat_template(self, messages, add_generation_prompt=False, enable_thinking=False):
        toks = self.hf_tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        if enable_thinking:
            toks += [151667] # <think>.
        return toks
    
    def get_eos_token_id(self):
        return self.hf_tokenizer.eos_token_id
    
    def get_pad_token_id(self):
        return 151643
    
    def decode(self, toks):
        return self.hf_tokenizer.decode(toks)
    
    def encode(self, text):
        return self.hf_tokenizer.encode(text)

def create_tokenizer(dir):
    config = json.loads(Path(dir+'tokenizer_config.json').read_text())
    config = {
        k: AddedToken(**v) if isinstance(v, dict) and str(k).endswith("token") else v for (k, v) in config.items()
    }
    config["added_tokens_decoder"] = {
        int(k): AddedToken(**v) for (k, v) in config.get("added_tokens_decoder", dict()).items()
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(dir+'tokenizer.json'), **config)
    tokenizer = Tokenizer(tokenizer)
    return tokenizer

if __name__ == "__main__":
    # Example usage
    tokenizer = create_tokenizer('/nfs/gcs/jaxconverted/Qwen3-0.6B/')
    print("Tokenizer created successfully.")
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Special tokens:", tokenizer.special_tokens_map)
    toks = tokenizer.encode("Hello, world!")
    print("Example tokenization of Hello, world!:", toks)
    print("Example decoding:", tokenizer.decode(toks))

    print("Pad ID:", tokenizer.pad_token_id) # 0

    for special in [
        '<|im_end|>', # eos token
        '<|im_start|>',
        '<|endoftext|>', # padding token
    ]:
        print(f"{special}:", tokenizer.encode(special))
    print('Decoding 0', tokenizer.decode([0])) # The 0 token is not padding, it's '!'.

    toks_msg = tokenizer.apply_chat_template([{"role": "user", "content": 'Hi'}], add_generation_prompt=True, enable_thinking=False)
    print("Example chat template (no thinking):", toks_msg)
    # [151644, <|im_start|>
    # 872, user
    # 198, newline
    # 13048, Hi
    # 151645, <|im_end|>
    # 198, newline
    # 151644, <|im_start|>
    # 77091, assistant
    # 198, newline
    # 151667, <think>
    # 271, double newline
    # 151668, </think>
    # 271] double newline
    print("Decoded chat template (no thinking):\n", tokenizer.decode(toks_msg))
    for i in range(len(toks_msg)):
        print(f"Token {i}:", toks_msg[i], "->", tokenizer.decode([toks_msg[i]]))

    toks_msg = tokenizer.apply_chat_template([{"role": "user", "content": 'Hi'}], add_generation_prompt=True, enable_thinking=True)
    print("Example chat template (with thinking):", toks_msg)
    print("Decoded chat template (with thinking):\n", tokenizer.decode(toks_msg))
    for i in range(len(toks_msg)):
        print(f"Token {i}:", toks_msg[i], "->", tokenizer.decode([toks_msg[i]]))
    # [151644, <|im_start|>
    # 872, user
    # 198, newline
    # 13048, Hi
    # 151645, <|im_end|>
    # 198, newline
    # 151644, <|im_start|>
    # 77091, assistant
    # 198, newline

    # Note: The model will keep generating, and eventually will output (151645, <|im_end|>). That's the end of generation.

