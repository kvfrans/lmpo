###############################
#
#  Structures for managing training of flax networks.
#
###############################

import flax
import jax
import functools
from typing import Any, Callable

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

# Contains model params and optimizer state.
class TrainState(flax.struct.PyTreeNode):
    rng: Any
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    params_ema: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    use_ema: bool = False

    @classmethod
    def create(cls, rng, model_def, model_input, tx, use_ema=False, **kwargs):
        params = model_def.init(rng, *model_input)['params']
        opt_state = tx.init(params)
        params_ema = params if use_ema else None
        return cls(
            rng=rng, step=1, apply_fn=model_def.apply, model_def=model_def, params=params, params_ema=params_ema,
            tx=tx, opt_state=opt_state, **kwargs,
        )
    
    @classmethod
    def create_with_params(cls, rng, model_def, params, tx, use_ema=False, **kwargs):
        opt_state = tx.init(params)
        params_ema = params if use_ema else None
        return cls(
            rng=rng, step=1, apply_fn=model_def.apply, model_def=model_def, params=params, params_ema=params_ema,
            tx=tx, opt_state=opt_state, **kwargs,
        )

    def call_model(self, *args, params=None, use_ema_params=False, **kwargs):
        if params is None:
            params = self.params if not use_ema_params else self.params_ema
        return self.apply_fn({"params": params}, *args, **kwargs)
    
    def update_ema(self, tau): # Tau should be close to 1, e.g. 0.999.
        new_params_ema = jax.tree_util.tree_map(
            lambda p, tp: p * (1-tau) + tp * tau, self.params, self.params_ema
        )
        return self.replace(params_ema=new_params_ema)

    # For pickling.
    def save(self):
        return {
            'params': self.params,
            'params_ema': self.params_ema,
            'opt_state': self.opt_state,
            'step': self.step,
        }
    
    def load(self, data):
        return self.replace(**data)