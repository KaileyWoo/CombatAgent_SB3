from gymnasium.envs.registration import register

register(
    id='MyCombatEnv-v0',
    entry_point='env.MyCombatEnv:MyCombatEnv',
    # Max number of steps per episode, using a `TimeLimitWrapper`
    #max_episode_steps=500,
)