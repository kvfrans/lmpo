### Helpers for sampling from models.
from lmpo.models.qwen3 import Qwen3Model, KVCache, count_left_padding
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial

from lmpo.utils.sharding import host_gather

def pad_and_collate(token_batch: list, pad_id: int = 0, force_length: int = None):
    max_len = max([len(x) for x in token_batch])
    max_len = max(host_gather(max_len)) if jax.process_count() > 1 else max_len
    if force_length is not None:
        if max_len > force_length:
            token_batch = [x[:force_length] for x in token_batch]
            print("Warning: Prompt tokens too long, truncating.")
        max_len = force_length
    return np.array([(max_len - len(x)) * [pad_id] + x for x in token_batch])

model_apply = None # Global variable to cache the JIT-compiled model application function.
def autoregressive_sample(model: Qwen3Model, params, prompt_tokens, num_generation_tokens, rng, temp=1, pad_id=0, data_shard=None, no_shard=None, force_answer_at=-1):
    """
    Samples tokens autoregressively, and can batch for performance.
    Args:
        prompt_tokens: An array of tokens, padded by `pad_id` on the LEFT. [batch, time].
        force_answer_at: If > 0, forces the insertion of an <answer> tag at (force_answer_at) tokens before the end of the generation.
    """
    global model_apply
    batch_size = prompt_tokens.shape[0]
    token_mask = jnp.where(prompt_tokens != pad_id, 1, 0).astype(jnp.int32)
    max_seq_len = prompt_tokens.shape[1] + num_generation_tokens

    cache = KVCache.create(model.num_layers, batch_size, max_seq_len, model.head_dim, model.kv_heads)
    cache = cache.replace(starts=count_left_padding(prompt_tokens, pad_id=pad_id))
    cache_sharding = KVCache.get_sharding(data_shard, no_shard)
    cache = jax.jit(lambda x: x, out_shardings=cache_sharding)(cache)

    if model_apply is None:
        @partial(jax.jit, out_shardings=(data_shard, cache_sharding))
        def model_apply(params, tokens, token_mask, cache):
            print("JIT compiling sampling for tokens of shape", tokens.shape)
            return model.apply({'params': params}, tokens, token_mask, cache=cache)

    # Fill cache with the prompt tokens.
    _, cache = model_apply(params, prompt_tokens[:, :-1], token_mask[:, :-1], cache=cache)
    sampled_token = prompt_tokens[:, -1]  # Start with the last token of the prompt.
    tokens_list = []

    max_samples = max_seq_len - prompt_tokens.shape[-1]
    for i in range(max_samples):
        next_token_mask = jnp.ones(sampled_token.shape, dtype=jnp.int32)
        logits, cache = model_apply(params, sampled_token[:, None], next_token_mask[:, None], cache=cache)
        logits = logits[:, 0, :]
        key, rng = jax.random.split(rng)
        if temp == 0:
            sampled_token = jnp.argmax(logits, axis=-1)
        else:
            sampled_token = jax.random.categorical(key, logits / temp, axis=-1) # [batch]

        # Yes, this is very ugly, even a sin. 
        # It's a helper flag to force insertion of an <answer> tag (force_answer_at) tokens before the end.
        if force_answer_at > 0:
            if i == max_samples - force_answer_at:
                sampled_token = jnp.ones_like(sampled_token) * 198 # \n
            elif i == max_samples - force_answer_at+1:
                sampled_token = jnp.ones_like(sampled_token) * 198 # \n
            elif i == max_samples - force_answer_at+2:
                sampled_token = jnp.ones_like(sampled_token) * 27 # <
            elif i == max_samples - force_answer_at+3:
                sampled_token = jnp.ones_like(sampled_token) * 9217 # answer
            elif i == max_samples - force_answer_at+4:
                sampled_token = jnp.ones_like(sampled_token) * 29 # />

        tokens_list.append(sampled_token)

    tokens = jnp.stack(tokens_list, axis=-1) # [batch, time]
    return tokens




######$###########################################
### Example of sampling an LLM to generate a poem.
##################################################
if __name__ == "__main__":
    import argparse
    from lmpo.models.qwen3 import create_model_from_ckpt
    from lmpo.utils.sharding import create_sharding, host_gather
    from lmpo.models.tokenizer import create_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='/nfs/gcs/jaxconverted/Qwen3-0.6B/')
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir

    model, params = create_model_from_ckpt(ckpt_dir)
    param_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', train_state_shape=params)
    params = jax.jit(lambda x: x, out_shardings=param_shard)(params)
    tokenizer = create_tokenizer(ckpt_dir)

    labels = ['cat', 'dog', 'bird', 'fish', 'elephant', 'tiger', 'lion', 'giraffe', 'zebra', 'monkey']
    poem_prompts = [f'Write a haiku about of {labels[np.random.randint(len(labels))]}' for _ in range(len(jax.local_devices()))]

    pad_id = 0
    token_list = [
        tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        for text in poem_prompts
    ]

    token_batch = pad_and_collate(token_list, pad_id=pad_id, force_length=256)
    print("Input tokens local:", token_batch.shape)
    token_batch = shard_data_fn(token_batch)
    print("Input tokens global:", token_batch.shape)
    num_generation_tokens = 32
    rng = jax.random.PRNGKey(0)
    tokens_out = autoregressive_sample(
        model, params, token_batch, rng=rng, num_generation_tokens=num_generation_tokens, pad_id=pad_id, data_shard=data_shard, no_shard=no_shard)
    tokens_out = host_gather(tokens_out)

    responses = [tokenizer.decode(row) for row in tokens_out]
    if jax.process_index() == 0:
        for i, text in enumerate(poem_prompts):
            print(f" ======= {text} =======")
            print(responses[i].split('<|im_end|>')[0])

        print("========= Full raw decoded tokens =========")
        print(tokenizer.decode(token_list[0] + tokens_out[0].tolist()))
        print('Total tokens shape', tokens_out.shape)
        print("=============")