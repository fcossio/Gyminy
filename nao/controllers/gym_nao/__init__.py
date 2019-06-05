from gym.envs.registration import register

register(
    id='gym_nao_standUp-v0',
    entry_point='gym_nao.envs:NaoStandUpEnv',
)

register(
    id='gym_nao_standUp-v1',
    entry_point='gym_nao.envs:NaoStandUp1Env',
)
