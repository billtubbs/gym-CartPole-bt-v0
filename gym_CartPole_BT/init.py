from gym.envs.registration import register

register(
    id='CartPole-BT-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
)