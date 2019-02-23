from gym.envs.registration import register

register(
    id='gym_nao_forwardWalk-v0',
    entry_point='gym_nao.envs:NaoForwardWalkEnv',
)
