from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax
import jax.numpy as jnp
import flax
import numpy as np

def create_sharding(shard_type, train_state_shape=None):
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('devices'))
    data_sharding = NamedSharding(mesh, PartitionSpec('devices'))
    no_shard = NamedSharding(mesh, PartitionSpec())
    num_hosts = jax.device_count() // len(jax.local_devices())

    if shard_type == 'dp':
        # Data-Parallelism.
        # - A full copy of params are on each device.
        # - Each device gets an independent slice of the batch.
        train_state_sharding = no_shard
    elif shard_type == 'fsdp':
        # Fully-Sharded Data Parallism.
        # - Each device gets an independent slice of the batch.
        # - Parameters are sharded among each device, along the largest axis.
        def shard_parameter(param):
            shape = param.shape
            all_nones = (None,) * param.ndim
            min_size_to_shard_mb = 4
            if np.prod(shape) * param.dtype.itemsize <= min_size_to_shard_mb * (2 ** 20):
                return all_nones
            if len(shape) == 2 and shape[1] > 150000:
                # Don't shard the output layer, because all-gather takes a long time.
                return all_nones
            
            # idx = np.argsort(shape)[::-1]
            idx = np.arange(len(shape))
            # This is neccessary to prevent numerical sharding issues. I cannot explain why.
            # But if params are sharded as (None, 'device'), it causes issues when doing
            # input @ params, when input is sharded with ['devices', None].


            for i in idx:
                if shape[i] % jax.device_count() == 0:
                    return all_nones[:i] + ('devices',) + all_nones[i+1:]
            print(f'Could not shard parameter of shape {shape}. Defaulting to full replication.')
            return all_nones
        train_state_sharding = jax.tree_util.tree_map(
            lambda spec: NamedSharding(mesh, PartitionSpec(*shard_parameter(spec))), 
            flax.linen.unbox(train_state_shape))

    # Shards a data along the first axis.
    # For single-host, this puts the data on the appropriate device.
    # For multi-host, call this with different data on each host. It will make a global array
    #     representing the data on all hosts, but only part will be addressable on this host.
    def shard_data(*args):
        def _shard_data(x):
            if jax.local_device_count() == jax.device_count():
                return jax.device_put(x, data_sharding)
            else:
                # Increases the first dimension by num_hosts. X is no longer fully addressable.
                x_shape = (x.shape[0] * num_hosts, *x.shape[1:])
                x = np.split(x, len(mesh.local_devices), axis = 0) # per device data, but on host
                x = jax.device_put(x, mesh.local_devices) # per device data, now on device
                return jax.make_array_from_single_device_arrays(x_shape, data_sharding, x)
        if len(args) == 1:
            return _shard_data(args[0])
        return jax.tree_map(_shard_data, args)
    
    # The first three are 'Sharding' objects which are pytrees.
    # The last two are helper functions for moving data between devices.
    return train_state_sharding, no_shard, data_sharding, shard_data

def host_gather(x):
    is_multi_host = len(jax.local_devices()) != len(jax.devices())
    return jax.experimental.multihost_utils.process_allgather(x) if is_multi_host else x