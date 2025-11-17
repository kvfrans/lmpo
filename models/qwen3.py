"""
Model structure for Qwen3. Taken from https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/model.py.
"""

import json
import glob
from safetensors import safe_open
import re
from functools import partial
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes, SegmentIds
from jax.experimental.pallas.ops.tpu.flash_attention import	flash_attention as pallas_flash_attention_tpu

import jax
import jax.numpy as jnp
import flax

import flax.linen as nn
import jax.numpy as jnp
import einops

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices=device_mesh, axis_names=('devices'))
data_sharding = PartitionSpec('devices')

def repeat_kv_heads(k, v, num_reps: int):
    return (
        einops.repeat(k, "b s h d -> b s (h r) d", r=num_reps),
        einops.repeat(v, "b s h d -> b s (h r) d", r=num_reps),
    )

@jax.remat
@partial(jax.shard_map, 
         mesh=mesh,
         in_specs=(data_sharding, data_sharding, data_sharding, data_sharding),
         out_specs=data_sharding,
         check_vma=False)
def flash_attention(q, k, v, token_mask):
    blocksize_q = 128
    blocksize_k = 128
    k, v = repeat_kv_heads(k, v, q.shape[2] // k.shape[2])
    query_length = q.shape[1]
    value_length = v.shape[1]
    # TPU implementation
    block_sizes = BlockSizes(
        block_q=min(blocksize_q, query_length),
        block_k_major=min(blocksize_k, value_length),
        block_k=min(blocksize_k, value_length),
        block_b=1,
        block_q_major_dkv=min(blocksize_q, query_length),
        block_k_major_dkv=min(blocksize_k, value_length),
        block_k_dkv=min(blocksize_k, value_length),
        block_q_dkv=min(blocksize_q, query_length),
        block_k_major_dq=min(blocksize_k, value_length),
        block_k_dq=min(blocksize_k, value_length),
        block_q_dq=min(blocksize_q, query_length),
    )

    segment_ids = SegmentIds(token_mask, token_mask)

    # NOTE: If running on GPU, try using the triton flash attention implementation:
    # from .flash_attention_triton import triton_flash_attention
    # attn = triton_flash_attention(
    #     q=q,
    #     k=k,
    #     v=v,
    #     bias=None,
    #     attention_mask=None,
    #     dropout_prob=0,
    #     causal=True if k.shape[1] == q.shape[1] else False,,
    #     dropout_seed=None,
    #     softmax_scale=1.0 / q.shape[-1]**0.5,
    # )
              
    return pallas_flash_attention_tpu(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        sm_scale=1.0 / q.shape[-1]**0.5,
        block_sizes=block_sizes,
        causal=True if k.shape[1] == q.shape[1] else False,
        segment_ids=segment_ids,
    ).transpose(0, 2, 1, 3).astype(jnp.bfloat16)

def rms_norm(x, gamma, eps):
    rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
    return jnp.astype(gamma * x / rms, jnp.bfloat16)

def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def count_left_padding(ids, pad_id=0):
    return jnp.sum(jnp.cumsum(ids != pad_id, axis=-1) == 0, axis=-1)

def length_minus_padding(token_mask):
    return jnp.sum(jnp.cumsum(jnp.flip(token_mask != 0, -1), axis=-1) > 0, -1)

def get_positions(token_mask):
    """Counts positions for segment ids."""
    def scan_fun(a, b):
        return ((a[0] + 1) * (a[1] == b[1]) + b[0], b[1])
    vals = (jnp.zeros_like(token_mask), token_mask)
    return jnp.array(jax.lax.associative_scan(scan_fun, vals, axis=-1)[0], dtype="int32")

def generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    rope_theta: float,
) -> tuple[jax.Array, jax.Array]:
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = rope_theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

class KVCache(flax.struct.PyTreeNode):
    k: list[jax.Array]
    v: list[jax.Array]
    length: int
    starts: jax.Array

    @classmethod
    def create(cls, num_layers, batch_size, max_seq_len, head_dim, kv_heads):
        k = [jnp.zeros((batch_size, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16) for _ in range(num_layers)]
        v = [jnp.zeros((batch_size, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16) for _ in range(num_layers)]
        length = 0
        starts = jnp.zeros((batch_size,), dtype=jnp.int32)
        return cls(k=k, v=v, length=length, starts=starts)

    @classmethod
    def get_sharding(cls, cache_shard, none_shard):
        # Yes, the type annotations are wrong, but this works nicely for jax sharding...
        return KVCache(k=cache_shard, v=cache_shard, length=none_shard, starts=cache_shard)

class Block(nn.Module):
    """ A standard transformer block. Has residual connection, self-attention, and a two-layer MLP. """
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    mlp_ffw_size: int
    eps: float = 1e-6
    use_flash_attn: bool = True

    @nn.compact
    def __call__(self, x, sin, cos, mask, token_mask, layer_id, cache=None):
        # We define the params out here in case we want to transform block_fwd with jax.remat.

        pre_gamma = self.param('pre_gamma', nn.initializers.constant(1.0), (self.hidden_size,))
        q_gamma = self.param('q_gamma', nn.initializers.constant(1.0), (self.head_dim,))
        k_gamma = self.param('k_gamma', nn.initializers.constant(1.0), (self.head_dim,))
        post_gamma = self.param('post_gamma', nn.initializers.constant(1.0), (self.hidden_size,))

        # =========================
        # === Self-Attention Block. 
        # =========================

        x_norm = rms_norm(x, pre_gamma, self.eps)
        
        # Calculate Q,K,V.
        q = nn.Dense(self.q_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        q = jnp.reshape(q, (q.shape[0], q.shape[1], self.q_heads, self.head_dim)) # (batch, seq_len, q_heads, head_dim)
        k = nn.Dense(self.kv_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        k = jnp.reshape(k, (k.shape[0], k.shape[1], self.kv_heads, self.head_dim))
        v = nn.Dense(self.kv_heads * self.head_dim, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        v = jnp.reshape(v, (v.shape[0], v.shape[1], self.kv_heads, self.head_dim))

        q = rms_norm(q, q_gamma, self.eps)
        q = apply_rotary_embedding(q, sin, cos)
        k = rms_norm(k, k_gamma, self.eps)
        k = apply_rotary_embedding(k, sin, cos)

        if cache is not None:
            k = jax.lax.dynamic_update_slice_in_dim(cache.k[layer_id], k, cache.length, axis=1)
            v = jax.lax.dynamic_update_slice_in_dim(cache.v[layer_id], v, cache.length, axis=1)

        if cache is None and self.use_flash_attn:
            # Flash attention for training.
            qkv = flash_attention(q,k,v,token_mask)
            qkv = jnp.reshape(qkv, (qkv.shape[0], qkv.shape[1], self.q_heads * self.head_dim))
        else:
            # Naive attention for inference.
            b, t, qh, d = q.shape # qh = 16
            _, T, kh, _ = k.shape # kh = 8
            q = jnp.reshape(q, (b, t, kh, qh // kh, d))
            qk = jnp.einsum("bthgd,bThd->btThg", q, k) * (d ** -0.5)
            qk = jnp.reshape(qk, (b, t, T, qh)) 
            qk = jnp.where(mask, qk, -1e30) # good
            attn = jax.nn.softmax(qk.astype(jnp.float32), axis=2) # on T dimension.
            attn = jnp.reshape(attn, (b, t, T, kh, qh // kh))
            qkv = jnp.einsum("btThg,bThd->bthgd", attn, v).astype(x.dtype)
            qkv = jnp.reshape(qkv, (b, t, qh*d))

        attn_x = nn.Dense(self.hidden_size, use_bias=False, dtype=jnp.bfloat16)(qkv)
        x = x + attn_x
        
        # =========================
        # === MLP Block. 
        # =========================
        x_norm = rms_norm(x, post_gamma, self.eps)
        g = nn.Dense(features=self.mlp_ffw_size, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        g = nn.silu(g)
        y = nn.Dense(features=self.mlp_ffw_size, use_bias=False, dtype=jnp.bfloat16)(x_norm)
        y = g * y
        mlp_x = nn.Dense(features=self.hidden_size, use_bias=False, dtype=jnp.bfloat16)(y)

        x = x + mlp_x
        return x, k, v

class Qwen3Model(nn.Module):
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    mlp_ffw_size: int
    num_layers: int
    rope_theta: int
    eps: float = 1e-6
    use_v_head: bool = False
    use_remat: bool = False

    @nn.compact
    def __call__(self, x, token_mask, cache = None, get_logits=True):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size)(x)
        x = x.astype(jnp.bfloat16)
        positions = get_positions(token_mask)
        if cache is not None:
            start_indices = jnp.where(cache.length != 0, cache.length - cache.starts, 0)
        else:
            start_indices = jnp.zeros((x.shape[0],), dtype=jnp.int32)
        positions = start_indices[:, None] + positions
        sin, cos = generate_pos_embeddings(positions, self.head_dim, self.rope_theta)
        sin, cos = sin.astype(jnp.bfloat16), cos.astype(jnp.bfloat16)

        # Attention Mask calculation: Do this once and resuse it.
        if cache is not None:
            T = cache.k[0].shape[1]
            time_idx = jnp.arange(0, T, dtype=jnp.int32)[None, :] # [1, seqlen]
            q_idx = jnp.where(token_mask != 0, 1, 0) # [B, seqlen] where tokens exist.
            incremental_pos = jnp.max(length_minus_padding(token_mask))
            k_idx = (time_idx >= cache.starts[:, None]) & (time_idx < (cache.length + incremental_pos).astype(jnp.int32))
            q_offset = cache.length
        else:
            T = x.shape[1]
            q_idx, k_idx = token_mask, token_mask
            q_offset = 0

        # Causal Attention Mask.
        mask = q_idx[:, :, None] & k_idx[:, None, :]
        mask = mask[:, None, :, :] # [B, 1, t, T]
        qk_size = (1, 1, x.shape[1], T)
        q_iota = jax.lax.broadcasted_iota(jnp.int32, qk_size, 2)
        k_iota = jax.lax.broadcasted_iota(jnp.int32, qk_size, 3)
        q_positions = q_iota + q_offset
        causal_mask = q_positions >= k_iota
        mask = jnp.logical_and(mask, causal_mask)
        mask = jnp.transpose(mask, (0, 2, 3, 1)) # [B, t, T, 1]

        BlockFn = Block if not self.use_remat else nn.remat(Block, static_argnums=6)
        for layer_id in range(self.num_layers):
            x, k, v = BlockFn(hidden_size=self.hidden_size, q_heads=self.q_heads, kv_heads=self.kv_heads, head_dim=self.head_dim, mlp_ffw_size=self.mlp_ffw_size, eps=self.eps)(x, sin, cos, mask, token_mask, layer_id, cache)
            if cache is not None:
                cache.k[layer_id] = k
                cache.v[layer_id] = v

        gamma_final = self.param('gamma_final', nn.initializers.constant(1.0), (self.hidden_size,))
        x = rms_norm(x, gamma_final, self.eps)
        if not self.use_v_head:
            if get_logits:
                logits = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.bfloat16)(x)
            else:
                logits = None
        else:
            logits = nn.Dense(1, use_bias=False, dtype=jnp.bfloat16)(x) # V-prediction head.

        if cache is not None:
            cache = cache.replace(length=cache.length + jnp.max(length_minus_padding(token_mask)))

        return logits, cache
    
###############################
##### Utils for loading models.
###############################

def create_model_from_hf(hf_dir: str):
    with open(hf_dir + "config.json") as f:
        cfg = json.load(f)
    model = Qwen3Model(
        hidden_size=cfg['hidden_size'],
        q_heads=cfg['num_attention_heads'],
        kv_heads=cfg['num_key_value_heads'],
        num_layers=cfg['num_hidden_layers'],
        head_dim=cfg['head_dim'],
        vocab_size=cfg['vocab_size'],
        mlp_ffw_size=cfg['intermediate_size'],
        eps=cfg['rms_norm_eps'],
        rope_theta=cfg['rope_theta']
    )
    tokens = jnp.ones((1,1), dtype=jnp.int32)
    idx = jnp.ones((1,1), dtype=jnp.int32)
    # params = model.init(jax.random.PRNGKey(0), tokens, idx)['params']
    params = jax.eval_shape(model.init, jax.random.PRNGKey(0), tokens, idx)['params']

    _HF_KEY_MAPPING = {
        r"model\.embed_tokens\.weight": "Embed_0.embedding",
        # attention projection weights
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"Block_\1.Dense_0.kernel",
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"Block_\1.Dense_1.kernel",
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"Block_\1.Dense_2.kernel",
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"Block_\1.Dense_3.kernel",
        # norms
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"Block_\1.q_gamma",
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"Block_\1.k_gamma",
        # layer norms (pre/post attention)
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"Block_\1.pre_gamma",
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"Block_\1.post_gamma",
        # mlp
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"Block_\1.Dense_4.kernel",
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"Block_\1.Dense_5.kernel",
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"Block_\1.Dense_6.kernel",
        r"model\.norm\.weight": "gamma_final",
        r"lm_head\.weight": "Dense_0.kernel",
    }

    def _torch_key_to_jax_key(source_key, custom_key_map: dict[str, str] | None = None):
        key_maps = dict(_HF_KEY_MAPPING, **(dict() if custom_key_map is None else custom_key_map))
        subs = [re.sub(pat, repl, source_key) for pat, repl in key_maps.items() if re.match(pat, source_key)]
        if len(subs) > 1:
            raise ValueError(f"More than 1 key matched: {subs}")
        else:
            return None if len(subs) == 0 else subs[0]

    torch_params = {}
    files = list(glob.glob(hf_dir+"*safetensors"))
    for file in files:
        with safe_open(file, framework="torch") as f:
            for key in f.keys():
                torch_params[key] = f.get_tensor(key)
                jax_key = _torch_key_to_jax_key(key)
                jax_key_list = jax_key.split('.')
                jax_param = params
                while len(jax_key_list) > 0:
                    jax_key = jax_key_list.pop(0)
                    if len(jax_key_list) == 0:
                        if 'kernel' in jax_key:
                            new_param = torch_params[key].float().T.numpy()
                        else:
                            new_param = torch_params[key].float().numpy()
                        assert new_param.shape == jax_param[jax_key].shape
                        jax_param[jax_key] = new_param
                    jax_param = jax_param[jax_key]
    
    return model, params

def create_model_from_ckpt(ckpt_dir: str, use_v_head: bool = False, use_remat: bool = False):
    from lmpo.utils.checkpoint import Checkpoint
    with open(ckpt_dir + "config.json") as f:
        cfg = json.load(f)
    model = Qwen3Model(
        hidden_size=cfg['hidden_size'],
        q_heads=cfg['num_attention_heads'],
        kv_heads=cfg['num_key_value_heads'],
        num_layers=cfg['num_hidden_layers'],
        head_dim=cfg['head_dim'],
        vocab_size=cfg['vocab_size'],
        mlp_ffw_size=cfg['intermediate_size'],
        eps=cfg['rms_norm_eps'],
        rope_theta=cfg['rope_theta'],
        use_v_head=use_v_head,
        use_remat=use_remat,
    )
    ckpt = Checkpoint(ckpt_dir + "params.pkl", parallel=False)
    params = ckpt.load_as_dict()['params']

    if use_remat:
        for key in list(params.keys()):
            if 'Block' in key:
                params['Checkpoint'+key] = params.pop(key)

    return model, params