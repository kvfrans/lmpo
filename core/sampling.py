### Helpers for sampling from models.
import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
import time
import tqdm

from lmpo.utils.sharding import host_gather, get_memory_usage
from lmpo.models.qwen3 import Qwen3Model, KVCache, count_left_padding

def pad_and_collate(token_batch: list, pad_id: int = 0, force_length: int = None):
    max_len = max([len(x) for x in token_batch])
    max_len = max(host_gather(max_len)) if jax.process_count() > 1 else max_len
    if force_length is not None:
        if max_len > force_length:
            token_batch = [x[:force_length] for x in token_batch]
            print(f"Warning: Prompt tokens too long, truncating. Empiricl max {max_len}, truncated to {force_length}")
        max_len = force_length
    return np.array([(max_len - len(x)) * [pad_id] + x for x in token_batch])

model_apply_prefill = None # Global variable to cache the JIT-compiled model application function.
model_apply_generate = None
def autoregressive_sample(model: Qwen3Model, params, prompt_tokens, num_generation_tokens, rng, temp=1, pad_id=0, data_shard=None, no_shard=None, prefill_batch_split=4, force_answer_at=-1, return_logprobs=False):
    """
    Samples tokens autoregressively, and can batch for performance.
    Args:
        prompt_tokens: An array of tokens, padded by `pad_id` on the LEFT. [batch, time].
        force_answer_at: If > 0, forces the insertion of an <answer> tag at (force_answer_at) tokens before the end of the generation.
    """
    global model_apply_prefill, model_apply_generate
    batch_size = prompt_tokens.shape[0]
    token_mask = jnp.where(prompt_tokens != pad_id, 1, 0).astype(jnp.int32)
    max_seq_len = prompt_tokens.shape[1] + num_generation_tokens

    cache_sharding = KVCache.get_sharding(data_shard, no_shard)
    params = jax.jit(lambda x: x, out_shardings=no_shard)(params)

    if model_apply_prefill is None:
        @partial(jax.jit, out_shardings=cache_sharding)
        def model_apply_prefill(params, tokens, token_mask):
            cache = KVCache.create(model.num_layers, tokens.shape[0], prompt_tokens.shape[1], model.head_dim, model.kv_heads)
            cache = cache.replace(starts=count_left_padding(tokens, pad_id=pad_id))
            _, cache = model.apply({'params': params}, tokens, token_mask, cache=cache, get_logits=False)
            return cache
        
    caches = []
    for i in range(prefill_batch_split):
        start_idx = (batch_size // prefill_batch_split) * i
        end_idx = (batch_size // prefill_batch_split) * (i + 1) if i < prefill_batch_split - 1 else batch_size
        cache = model_apply_prefill(params, prompt_tokens[start_idx:end_idx], token_mask[start_idx:end_idx])
        caches.append(cache)
            
    starts = jnp.concatenate([c.starts for c in caches], axis=0)
    cache = KVCache(k=[], v=[], length=caches[0].length, starts=starts)

    @partial(jax.jit, out_shardings=data_shard)
    def concat_kv(k, v):
        k = jnp.concatenate(k, axis=0)
        v = jnp.concatenate(v, axis=0)
        k = jnp.pad(k, ((0,0),(0,max_seq_len - k.shape[1]),(0,0),(0,0)), constant_values=0)
        v = jnp.pad(v, ((0,0),(0,max_seq_len - v.shape[1]),(0,0),(0,0)), constant_values=0)
        return k, v
    for i in range(len(caches[0].k)-1, -1, -1):
        k, v = concat_kv([c.k[i] for c in caches], [c.v[i] for c in caches])
        cache.k.append(k)
        cache.v.append(v)
        for c in caches:
            del c.k[i]
            del c.v[i]
        # print("Memory usage after concat cache:", get_memory_usage(), "GB")
    del caches
    cache = cache.replace(k=cache.k[::-1], v=cache.v[::-1]) # Reverse the list.
    reshape_fn = jax.jit(lambda x: x, out_shardings=cache_sharding, donate_argnums=0)
    cache = reshape_fn(cache) # Reshard the cache.
    print("Memory usage after reshard cache:", get_memory_usage(), "GB")

    if model_apply_generate is None:
        @partial(jax.jit, out_shardings=(no_shard, no_shard, cache_sharding), donate_argnums=3)
        def model_apply_generate(params, tokens, token_mask, cache, key):
            logits, cache = model.apply({'params': params}, tokens, token_mask, cache=cache, get_logits=True)
            logits = logits[:, 0, :]
            logprobs = jax.nn.log_softmax(logits / temp, axis=-1)
            sampled_token = jax.random.categorical(key, logits/temp, axis=-1)
            sampled_logprobs = jnp.sum(logprobs * jax.nn.one_hot(sampled_token, logits.shape[-1]), axis=-1)
            return sampled_token, sampled_logprobs, cache

    sampled_token = prompt_tokens[:, -1]  # Start with the last token of the prompt.
    tokens_list = []
    logprobs_list = []

    max_samples = max_seq_len - prompt_tokens.shape[-1]
    for i in tqdm.tqdm(range(max_samples)):
        next_token_mask = jnp.ones(sampled_token.shape, dtype=jnp.int32)
        key, rng = jax.random.split(rng)

        sampled_token, sampled_logprobs, cache = model_apply_generate(params, sampled_token[:, None], next_token_mask[:, None], cache=cache, key=key)

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
        logprobs_list.append(sampled_logprobs)

    tokens = jnp.stack(tokens_list, axis=-1) # [batch, time]
    logprobs = jnp.stack(logprobs_list, axis=-1) # [batch, time]
    logprobs = jax.device_get(logprobs)
    if return_logprobs:
        return tokens, logprobs
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
    parser.add_argument('--ckpt-dir', type=str, default='/nfs/gcs/jaxconverted/Qwen3-0.6B/')
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