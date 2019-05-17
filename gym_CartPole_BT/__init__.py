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
        'description': "Basic cart-pendulum system with large disturbances",
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
        'description': "Basic cart-pendulum system with large variance in "
                       "initial state",
        'initial_state_variance': 'high'
    }
)

register(
    id='CartPole-BT-dL-vL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with small disturbances"
                       "and small variance in initial state",
        'initial_state_variance': 'low',
        'disturbances': 'low'
    }
)

register(
    id='CartPole-BT-dH-vH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with large disturbances"
                       "and large variance in initial state",
        'initial_state_variance': 'high',
        'disturbances': 'high'
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

register(
    id='CartPole-BT-m2-dL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal and small disturbances",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.]),
        'disturbances': 'low'
    }
)

register(
    id='CartPole-BT-m2-dH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal and large disturbances",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.]),
        'disturbances': 'high'
    }
)

