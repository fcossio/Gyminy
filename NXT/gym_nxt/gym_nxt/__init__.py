from gym.envs.registration import register

register(
    id='nxt-v0',
    entry_point='gym_nxt.envs:NXTSimpleEnv',
)
