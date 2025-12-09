

def create_env(env_name, tokenizer):
    env_name = env_name.lower()
    if env_name == 'poem':
        from lmpo.envs.poem_length import PoemLengthEnv
        env = PoemLengthEnv(tokenizer)
    elif env_name == 'countdown':
        from lmpo.envs.countdown import CountdownEnv
        env = CountdownEnv(tokenizer)
    elif env_name == 'countdown6':
        from lmpo.envs.countdown import CountdownEnvSix
        env = CountdownEnvSix(tokenizer)
    else:
        from lmpo.envs.math_base import MathEnv
        env = MathEnv(tokenizer, env_name)
    return env