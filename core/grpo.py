import jax.numpy as jnp
import jax
import numpy as np
import tqdm
import optax
from functools import partial
import wandb
import ml_collections
import sys
import time
import shutil
from absl import app, flags

try: # If you like to use these helpers, you can.
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir('/home/kvfrans/jax-cache')
    from localutils.debugger import enable_debug
    enable_debug()
except:
    pass

from lmpo.models.qwen3 import create_model_from_ckpt
from lmpo.utils.configs import define_flag_dict
from lmpo.utils.wandb import setup_wandb
from lmpo.envs.env_creator import create_env
from lmpo.utils.sharding import create_sharding, host_gather, get_memory_usage, get_local_slice
from lmpo.utils.train_state import TrainState
from lmpo.models.tokenizer import create_tokenizer
from lmpo.utils.checkpoint import Checkpoint
from lmpo.core.sampling import pad_and_collate, autoregressive_sample
from lmpo.core.eval import eval_model

config = ml_collections.ConfigDict({
    'wandb_project': "lmpo",
    'wandb_name': 'lmpo-run',
    'wandb_group': 'Default',
    'model_dir': '/gcs/jaxconverted/Qwen3-1.7B/',
    'save_dir': "",
    'save_interval': 20,
    # env settings.
    'env_name': 'poem', # (poem, gsm8k, countdown)
    'num_generation_tokens': -1, # -1 = use default from env.
    'prompt_length': 256,
    'force_answer_at': -1, # -1 = use default from env.
    'test_env_name': '',
    'test_interval': 10,
    # sampling settings.
    'prefill_batch_split': 4,
    'inference_batch_per_device': 64, # Set this to the maximum until OOM. Should not affect results.
    # training settings.
    'groups_per_batch': 256, # global batch = groups_per_batch * group_size
    'ppo_minibatch': 64,
    'group_size': 8, # GRPO group size.
    'do_group_normalization': 1,
    'do_global_normalization': 0,
    'do_group_filter': 1, # Filter for groups with all advantages == 0.
    'do_clip_advantages': 0, # Clip advantages to be positive.
    'do_inference_logprobs': 0, # Use inference-time logprobs for importance sampling ratio.
    'do_mask_inference_ratio': 0, # Mask out tokens with a bad inference/recompute ratio.
    'lr': 1e-6,
    'clip_epsilon': 0.2,
    'do_ppo_all_clip': 0, # Clips both sides of ratio.
    'entropy_coef': 0.001,
    'kl_coef': 0.001,
    'weight_decay': 1e-2,
})
define_flag_dict(config)
FLAGS = flags.FLAGS
FLAGS(sys.argv)
if jax.process_index() == 0:
    setup_wandb(FLAGS.flag_values_dict(), project=FLAGS.wandb_project, name=FLAGS.env_name+'-'+FLAGS.wandb_name, group=FLAGS.wandb_group)
    rollouts_list = []

host_id = jax.process_index()
                                          
ckpt_dir = FLAGS.model_dir
model, params = create_model_from_ckpt(ckpt_dir)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(FLAGS.lr, b1=0.9, b2=0.95, weight_decay=FLAGS.weight_decay),
)
rng = jax.random.PRNGKey(0)
print("Memory usage pre-init:", get_memory_usage(), "GB")
init_fn = partial(TrainState.create_with_params, model_def=model, tx=tx, use_ema=False)
train_state_shape = jax.eval_shape(init_fn, rng=rng, params=params)
train_state_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', train_state_shape)
train_state = jax.jit(lambda r, p: init_fn(rng=r, params=p), out_shardings=train_state_shard)(rng, params)
print("Memory usage train_state:", get_memory_usage(), "GB")

jax.debug.visualize_array_sharding(train_state.params['Block_0']['Dense_0']['kernel'])
tokenizer = create_tokenizer(ckpt_dir)
pad_id = tokenizer.get_pad_token_id()

env = create_env(FLAGS.env_name, tokenizer)
env_test = create_env(FLAGS.test_env_name, tokenizer) if FLAGS.test_env_name != '' else None

if FLAGS.num_generation_tokens == -1:
    FLAGS.num_generation_tokens = env.tokens_per_action
if FLAGS.force_answer_at == -1:
    FLAGS.force_answer_at = env.force_answer_at
np.random.seed(jax.process_index())
env_num_tasks = env.num_tasks if env.num_tasks != -1 else 1000000
env_task_idx = 0

@jax.jit
def get_logprobs(train_state: TrainState, token_batch, mask):
    print("JIT compiling logprob function for token_batch of shape", token_batch.shape)
    text_input, text_target = token_batch[:, :-1], token_batch[:, 1:]
    mask = mask[:, 1:]
    token_mask = jnp.where(text_input != pad_id, 1, 0).astype(jnp.int32)
    logits, _ = train_state.call_model(text_input, token_mask, cache=None)
    logprobs = jax.nn.log_softmax(logits, axis=-1) # [batch, time, vocab_size]
    logprobs = jnp.sum(logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
    return logprobs

@partial(jax.jit, out_shardings=(train_state_shard, None))
def update(train_state: TrainState, token_batch, mask_origin, advantages, recalc_logprobs, inference_logprobs):
    print("JIT compiling update function for token_batch of shape", token_batch.shape)
    text_input, text_target = token_batch[:, :-1], token_batch[:, 1:]
    inference_logprobs = inference_logprobs[:, 1:]
    mask_origin = mask_origin[:, 1:]
    is_max_tokens = (mask_origin[:, -1] == True)
    token_mask = jnp.where(text_input != pad_id, 1, 0).astype(jnp.int32)
    def loss_fn(grad_params):
        logits, _ = train_state.call_model(text_input, token_mask, cache=None, params=grad_params)
        all_logprobs = jax.nn.log_softmax(logits) # [batch, time, vocab_size]
        token_logprobs = jnp.sum(all_logprobs * jax.nn.one_hot(text_target, logits.shape[-1]), axis=-1)
        entropy = -jnp.sum(jax.nn.softmax(logits) * all_logprobs, axis=-1)

        old_logprobs = inference_logprobs if FLAGS.do_inference_logprobs else recalc_logprobs
        ratio_recompute_inference = jnp.exp(inference_logprobs - recalc_logprobs)
        mask = mask_origin
        if FLAGS.do_mask_inference_ratio:
            mask = mask * (jnp.abs(ratio_recompute_inference) - 1 < 1.0)

        # PPO loss.
        logratio = token_logprobs - old_logprobs
        ratio = jnp.exp(logratio)
        if FLAGS.do_ppo_all_clip:
            pg_loss = -advantages[:, None] * jnp.clip(ratio, 1 - FLAGS.clip_epsilon, 1 + FLAGS.clip_epsilon)
        else:
            pg_loss1 = -advantages[:, None] * ratio
            pg_loss2 = -advantages[:, None] * jnp.clip(ratio, 1 - FLAGS.clip_epsilon, 1 + FLAGS.clip_epsilon)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2)

        # Metrics
        avg_over_mask = lambda x : jnp.sum(x * mask) / jnp.sum(mask)
        importance_ratio = avg_over_mask(ratio)
        importance_ratio_mag = avg_over_mask(jnp.abs(1 - ratio))
        approx_kl = avg_over_mask((ratio - 1) - logratio)
        entropy_avg = avg_over_mask(entropy)
        clip_fracs = avg_over_mask(jnp.abs(ratio - 1.0) > FLAGS.clip_epsilon)
        logprob_of_token = avg_over_mask(-token_logprobs)
        inference_recompute_kl = avg_over_mask((ratio_recompute_inference - 1) - (inference_logprobs - recalc_logprobs))
        inference_recompute_prob_diff = jnp.abs(jnp.exp(inference_logprobs) - jnp.exp(recalc_logprobs)) * mask

        loss_pg = jnp.mean(pg_loss * mask)
        loss_ent = -jnp.mean(entropy_avg * mask) * FLAGS.entropy_coef
        loss = loss_pg + loss_ent
        return loss, {
            'loss': loss,
            'loss_pg': loss_pg,
            'loss_ent': loss_ent,
            'advantages': jnp.mean(advantages),
            'advantages_magnitude': jnp.mean(jnp.abs(advantages)),
            'nonzero_advantages': jnp.mean(advantages != 0),
            'entropy': entropy_avg,
            'approx_kl': approx_kl,
            'inference_recompute/kl': inference_recompute_kl,
            'inference_recompute/prob_diff_mean': jnp.mean(inference_recompute_prob_diff),
            'inference_recompute/prob_diff_99quantile': jnp.quantile(inference_recompute_prob_diff, 0.99),
            'inference_recompute/prob_diff_max': jnp.max(inference_recompute_prob_diff),
            'clip_fraction': clip_fracs,
            'logprob_of_token': logprob_of_token,
            'importance_ratio/mean': importance_ratio,
            'importance_ratio/magnitude': importance_ratio_mag,
            'importance_ratio/99quantile': jnp.quantile(ratio * mask, 0.99),
            'importance_ratio/max': jnp.max(ratio * mask),
            'importance_ratio/min': jnp.min(ratio * mask),
            'trained_tokens_per_seq': jnp.mean(jnp.sum(mask, axis=-1)),
            'is_max_tokens': jnp.mean(is_max_tokens),
        }
    grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
    updates, opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    info['grad_norm'] = optax.global_norm(grads)
    info['update_norm'] = optax.global_norm(updates)
    info['param_norm'] = optax.global_norm(new_params)
    train_state = train_state.replace(
        params=new_params,
        opt_state=opt_state,
        step=train_state.step + 1,
    )
    return train_state, info

rollout_batch_size = jax.local_device_count() * FLAGS.inference_batch_per_device
assert rollout_batch_size % FLAGS.group_size == 0
rng = jax.random.PRNGKey(jax.process_index())
total_rollouts = 0

for i in tqdm.tqdm(range(10000)):

    # Fill this global on-policy buffer with groups that have A != 0.
    buffer_tokens = []
    buffer_inference_logprobs = []
    buffer_advantages = []
    env_infos_history = {}
    env_infos_history['return'] = []
    num_rollout_iters = 0
    rollout_start_time = time.time()
    while len(buffer_tokens) < FLAGS.groups_per_batch:
        num_rollout_iters += 1
        total_rollouts += rollout_batch_size * jax.process_count()
        env_states, env_tokens = [], []
        for _ in range(rollout_batch_size // FLAGS.group_size):
            env_state, output_tokens = env.reset(min(env_task_idx + jax.process_index(), env_num_tasks-1))
            env_task_idx += jax.process_count()
            env_task_idx = env_task_idx % env_num_tasks
            for _ in range(FLAGS.group_size):
                env_states.append(env_state)
                env_tokens.append(output_tokens)

        prompt_tokens = pad_and_collate(env_tokens, pad_id=pad_id, force_length=FLAGS.prompt_length)
        prompt_tokens = shard_data_fn(prompt_tokens)
        num_generation_tokens = FLAGS.num_generation_tokens
        rng, key = jax.random.split(rng)
        action_tokens, action_logprobs = autoregressive_sample(
            train_state.model_def, train_state.params, prompt_tokens, rng=key, num_generation_tokens=num_generation_tokens, 
            pad_id=pad_id, data_shard=data_shard, no_shard=no_shard, force_answer_at=FLAGS.force_answer_at, 
            prefill_batch_split=FLAGS.prefill_batch_split, return_logprobs=True
        )
        prompt_tokens = host_gather(prompt_tokens)
        action_tokens = host_gather(action_tokens)
        all_tokens = jnp.concatenate([prompt_tokens, action_tokens], axis=-1)
        all_logprobs = jnp.concatenate([jnp.zeros_like(prompt_tokens), action_logprobs], axis=-1)

        action_tokens_local = get_local_slice(action_tokens, data_shard.mesh)
        new_states, _, returns_local, dones, env_infos = env.step_list(env_states, [t.tolist() for t in action_tokens_local])
        assert dones[0] # Only supports bandit envs for now.
        returns_local = np.array(returns_local)
        returns = host_gather(shard_data_fn(returns_local))
        for k, v in env_infos.items():
            if k not in env_infos_history:
                env_infos_history[k] = []
            v_global = host_gather(shard_data_fn(np.array(v)))
            env_infos_history[k] += v_global.tolist()
        env_infos_history['return'] += returns.tolist()


        mask_size = prompt_tokens.shape[-1]

        # Advantage calculation.
        returns = jnp.reshape(returns, (-1, FLAGS.group_size))
        advantages = returns
        if FLAGS.do_group_normalization:
            group_mean = np.mean(advantages, axis=-1)
            group_std = np.std(advantages, axis=-1) + 1e-8
            advantages = (advantages - group_mean[:, None]) / group_std[:, None]
        if FLAGS.do_global_normalization:
            global_mean = np.mean(advantages)
            global_std = np.std(advantages) + 1e-8
            advantages = (advantages - global_mean) / global_std
        if FLAGS.do_clip_advantages:
            advantages = np.clip(advantages, a_min=0, a_max=None)
        advantages_grouped = advantages # [batch_size // group_size, group_size]
        all_tokens_grouped = all_tokens.reshape(-1, FLAGS.group_size, all_tokens.shape[-1])
        all_logprobs_grouped = all_logprobs.reshape(-1, FLAGS.group_size, all_logprobs.shape[-1])

        for group_idx in range(advantages_grouped.shape[0]):
            if np.all(advantages_grouped[group_idx, :] == 0) and FLAGS.do_group_filter:
                continue
            else:
                buffer_tokens.append(all_tokens_grouped[group_idx, :])
                buffer_advantages.append(advantages_grouped[group_idx, :])
                buffer_inference_logprobs.append(all_logprobs_grouped[group_idx, :])
        print(f"Buffer size: {len(buffer_tokens) * FLAGS.group_size}. Return avg: {np.mean(returns)}")
        if jax.process_index() == 0:
            print(env.render(new_states[0]))

    rollout_total_time = time.time() - rollout_start_time

    def ppo_shard(x):
        """Helper function that takes a local buffer, shards across devices, then splits into PPO minibatches."""
        host_id = jax.process_index()
        host_slice = FLAGS.ppo_minibatch // jax.process_count()
        x = jnp.reshape(x, (FLAGS.ppo_minibatch, -1, *x.shape[1:]))
        x = x[host_id * host_slice : (host_id + 1) * host_slice, :]
        x = shard_data_fn(x)
        return x # [ppo_minibatch, num_minibatches (j), ...] where first dim is sharded.

    # The buffer is syncronized among hosts.
    tokens_all = jnp.concatenate(buffer_tokens, axis=0)
    advantages = jnp.concatenate(buffer_advantages, axis=0)
    inference_logprobs_all = jnp.concatenate(buffer_inference_logprobs, axis=0)
    global_batch_size = FLAGS.groups_per_batch * FLAGS.group_size
    print(f"Clipping total buffer of {tokens_all.shape[0]} to global batch size {global_batch_size}.")
    tokens_all = tokens_all[:global_batch_size]
    advantages = advantages[:global_batch_size]
    inference_logprobs_all = inference_logprobs_all[:global_batch_size]

    # Mask = False for all prompt tokens, and tokens after <|im_end|> token.
    mask = (jnp.arange(tokens_all.shape[-1]) >= mask_size - 1)[None, :]
    eos_idx = jnp.argmax(tokens_all[:, mask_size:] == tokenizer.get_eos_token_id(), axis=-1)
    eos_idx = jnp.where(eos_idx == 0, tokens_all.shape[-1], eos_idx)
    mask = mask & (jnp.arange(tokens_all.shape[-1])[None, :] <= eos_idx[:, None] + mask_size)

    tokens_all_minibatch = ppo_shard(tokens_all)
    advantages_minibatch = ppo_shard(advantages)
    inference_logprobs_all_minibatch = ppo_shard(inference_logprobs_all)
    mask_minibatch = ppo_shard(mask)

    # First, we do a forward pass to get prior logprobs for each token.
    logprobs_list = []
    for j in range(global_batch_size // FLAGS.ppo_minibatch):
        logprobs_minibatch = get_logprobs(train_state, tokens_all_minibatch[:, j], mask_minibatch[:, j])
        logprobs_list.append(logprobs_minibatch)
    logprobs_all_minibatch = jnp.stack(logprobs_list, axis=1)

    # Then, the training loop.
    update_time_start = time.time()
    for j in range(global_batch_size // FLAGS.ppo_minibatch):
        train_state, info = update(train_state, tokens_all_minibatch[:, j], mask_minibatch[:, j], advantages_minibatch[:, j], 
                                   logprobs_all_minibatch[:, j], inference_logprobs_all_minibatch[:, j])
        info = jax.device_get(info)
        info['output_tokens'] = eos_idx
        info = jax.tree.map(lambda x: np.array(x), info)
        info = jax.tree.map(lambda x: x.mean(), info)
        info['total_rollouts'] = total_rollouts
        if env.num_tasks != -1:
            info['env_epochs'] = total_rollouts / env_num_tasks
        info['rollout_iters_per_update'] = num_rollout_iters
        info['global_step'] = i
        info['times/time_per_inference_iteration'] = rollout_total_time / num_rollout_iters
        info['times/time_per_rollout'] = rollout_total_time / (num_rollout_iters * rollout_batch_size * jax.host_count())
        info['times/time_per_effective_rollout'] = rollout_total_time / global_batch_size
        info['times/total_time_rollouts'] = rollout_total_time
        info['times/total_time_update'] = time.time() - update_time_start
        info['effective_rollout_ratio'] = global_batch_size / (rollout_batch_size * jax.host_count() * num_rollout_iters)
        info['minibatches_per_global_step'] = global_batch_size // FLAGS.ppo_minibatch
        for k, v in env_infos_history.items():
            info['env/'+k] = np.mean(v)
        if 'action_length' in env_infos_history:
            info['env/is_max_tokens'] = np.mean(np.array(env_infos_history['action_length']) >= env.tokens_per_action)
        if jax.process_index() == 0:
            rollouts_list.append([i, env.render(new_states[0]), returns_local[0]])
            if i % 100 == 0 and j == 0:
                rollouts_table = wandb.Table(data=rollouts_list, columns=["step", "text", "reward"])
                info['rollouts_table'] = rollouts_table
            if j == global_batch_size // FLAGS.ppo_minibatch - 1:
                print(f'=================== Iter {i} ===================')
                for k, v in info.items():
                    if k not in ['rollouts_table']:
                        print(f"{k}: {v}")
            wandb.log(info)

    if i % FLAGS.test_interval == 0 and env_test is not None:
        _, test_env_history = eval_model(
            model=train_state.model_def,
            params=train_state.params,
            env=env_test,
            num_generation_tokens=FLAGS.num_generation_tokens,
            force_answer_at=FLAGS.force_answer_at,
            prompt_length=FLAGS.prompt_length,
            inference_batch_per_device=FLAGS.inference_batch_per_device,
            pad_id=pad_id,
            shard_data_fn=shard_data_fn,
            no_shard=no_shard,
            data_shard=data_shard,
            num_epochs=1,
        )
        test_info = {f'test_env/{k}': np.mean(v) for k, v in test_env_history.items()}
        if jax.process_index() == 0:
            wandb.log(test_info, commit=False)

    # This only saves the params. If you want to save the optimizer, gather the whole train_state.
    if i % FLAGS.save_interval == 0 and FLAGS.save_dir != "":
        params_gather = host_gather(train_state.params)
        if jax.process_index() == 0:
            step_dir = FLAGS.save_dir + '/step' + str(i ) + '/'
            cp = Checkpoint(step_dir + 'params.pkl', parallel=False)
            cp.params = params_gather
            cp.save()
            del cp
            shutil.copy(FLAGS.model_dir + 'config.json', step_dir + 'config.json')
            shutil.copy(FLAGS.model_dir + 'tokenizer_config.json', step_dir + 'tokenizer_config.json')
            shutil.copy(FLAGS.model_dir + 'tokenizer.json', step_dir + 'tokenizer.json')

        del params_gather