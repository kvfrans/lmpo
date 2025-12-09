from absl import app, flags

def define_flag_dict(config):
    for k, v in config.items():
        if type(v) is str:
            flags.DEFINE_string(k, v, f"Configuration for {k}")
        elif type(v) is int:
            flags.DEFINE_integer(k, v, f"Configuration for {k}")
        elif type(v) is float:
            flags.DEFINE_float(k, v, f"Configuration for {k}")