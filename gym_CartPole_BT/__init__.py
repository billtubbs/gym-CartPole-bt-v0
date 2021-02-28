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
        'description': "Basic cart-pendulum system with low random disturbance",
        'disturbances': 'low'
    }
)

register(
    id='CartPole-BT-dH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with high random disturbance",
        'disturbances': 'high'
    }
)

register(
    id='CartPole-BT-vL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with low variance in "
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
    id='CartPole-BT-dL-vL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with low random "
                       "disturbance and low variance in initial state",
        'initial_state_variance': 'low',
        'disturbances': 'low'
    }
)

register(
    id='CartPole-BT-dH-vH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with high random "
                       "disturbance and high variance in initial state",
        'initial_state_variance': 'high',
        'disturbances': 'high'
    }
)

register(
    id='CartPole-BT-p2-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle)",
    }
)

register(
    id='CartPole-BT-p2-dL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "small disturbance",
        'disturbances': 'low'
    }
)

register(
    id='CartPole-BT-p2-dH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "high random disturbance",
        'disturbances': 'high'
    }
)

register(
    id='CartPole-BT-p2-vL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "low variance in initial state",
        'initial_state_variance': 'low'
    }
)

register(
    id='CartPole-BT-p2-vH-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "high variance in initial state",
        'initial_state_variance': 'high'
    }
)

register(
    id='CartPole-BT-m2-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal state",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.])
    }
)

register(
    id='CartPole-BT-m2-dL-v0',
    entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
    kwargs={
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal and small random disturbance",
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
                       "distance -2 from goal and high random disturbance",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.]),
        'disturbances': 'high'
    }
)