def create_env(env_name, tokenizer):
    env_name = env_name.lower()
    if env_name == 'poem':
        from lmpo.envs.poem_length import PoemLengthEnv
        env = PoemLengthEnv(tokenizer)
    elif env_name == 'gsm8k':
        from lmpo.envs.gsm8k import GSM8KEnv
        env = GSM8KEnv(tokenizer)
    elif env_name == 'gsm8k-test':
        env = GSM8KEnv(tokenizer, train=False)
    elif env_name == 'countdown':
        from lmpo.envs.countdown import CountdownEnv
        env = CountdownEnv(tokenizer)
    elif env_name == 'deepscaler':
        from lmpo.envs.deepscaler import DeepscalerEnv
        env = DeepscalerEnv(tokenizer)
    elif env_name == 'aime':
        from lmpo.envs.aime import AimeEnv
        env = AimeEnv(tokenizer)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    return env