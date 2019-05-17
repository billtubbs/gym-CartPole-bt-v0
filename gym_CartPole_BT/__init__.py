from gym.envs.registration import register
import numpy as np

register(
    id='CartPole-BT-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={'description': "Basic cart-pendulum system"}
)

register(
    id='CartPole-BT-dL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with small disturbances",
        'disturbances': 'low'
    }
)

register(
    id='CartPole-BT-dH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with high disturbances",
        'disturbances': 'high'
    }
)

register(
    id='CartPole-BT-vL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with small variance in "
                       "initial state",
        'initial_state_variance': 'low'
    }
)

register(
    id='CartPole-BT-vH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with high variance in "
                       "initial state",
        'initial_state_variance': 'high'
    }
)

register(
    id='CartPole-BT-m2-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.])
    }
)

