from gym.envs.registration import register
import numpy as np

# Symbols used to label environment variants
# 'p' - partially-observed states: ['p2']
# 'x' - initial cart x-position -2 from goal: ['x2']
# 'd' - state disturbance: ['dL', 'dH']
# 'v' - variance in initial states: ['vL', 'vH']
# 'm' - measurement noise: ['mL', 'mH']
# 'v' - (at the end) version number: e.g. 'v0'

# Define a variety of different environment configurations
env_configs = {
    'CartPole-BT-v0': {
        'description': "Basic cart-pendulum system"
    },
    'CartPole-BT-dL-v0': {
        'description': "Basic cart-pendulum system with low random disturbance",
        'disturbances': 'low'
    },
    'CartPole-BT-dH-v0': {
        'description': "Basic cart-pendulum system with high random disturbance",
        'disturbances': 'high'
    },
    'CartPole-BT-vL-v0': {
        'description': "Basic cart-pendulum system with low variance in "
                       "initial state",
        'initial_state_variance': 'low'
    },
    'CartPole-BT-vH-v0': {
        'description': "Basic cart-pendulum system with high variance in "
                       "initial state",
        'initial_state_variance': 'high'
    },
    'CartPole-BT-dL-vL-v0': {
        'description': "Basic cart-pendulum system with low random "
                       "disturbance and low variance in initial state",
        'initial_state_variance': 'low',
        'disturbances': 'low'
    },
    'CartPole-BT-dH-vH-v0': {
        'description': "Basic cart-pendulum system with high random "
                       "disturbance and high variance in initial state",
        'initial_state_variance': 'high',
        'disturbances': 'high'
    },
    'CartPole-BT-p2-v0': {
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle)",
        'output_matrix': ((1, 0, 0, 0), (0, 0, 1, 0))
    },
    'CartPole-BT-p2-dL-v0': {
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "low disturbance",
        'output_matrix': ((1, 0, 0, 0), (0, 0, 1, 0)),
        'disturbances': 'low'
    },
    'CartPole-BT-p2-dH-v0': {
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "high random disturbance",
        'output_matrix': ((1, 0, 0, 0), (0, 0, 1, 0)),
        'disturbances': 'high'
    },
    'CartPole-BT-p2-vL-v0': {
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "low variance in initial state",
        'output_matrix': ((1, 0, 0, 0), (0, 0, 1, 0)),
        'initial_state_variance': 'low'
    },
    'CartPole-BT-p2-vH-v0': {
        'description': "Basic cart-pendulum system with 2 of 4 states "
                       "measured (cart x-position and pole angle) and "
                       "high variance in initial state",
        'output_matrix': ((1, 0, 0, 0), (0, 0, 1, 0)),
        'initial_state_variance': 'high'
    },
    'CartPole-BT-x2-v0': {
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal state",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.])
    },
    'CartPole-BT-x2-dL-v0': {
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal and low random disturbance",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.]),
        'disturbances': 'low'
    },
    'CartPole-BT-x2-dH-v0': {
        'description': "Basic cart-pendulum system with initial state "
                       "distance -2 from goal and high random disturbance",
        'initial_state': np.array([-1., 0., np.pi, 0.]),
        'goal_state': np.array([1., 0., np.pi, 0.]),
        'disturbances': 'high'
    }
}

# Register the environments with OpenAI Gym
for id, env_config in env_configs.items():
    register(
        id=id, 
        entry_point='gym_CartPole_BT.envs:CartPoleBTEnv',
        kwargs={'env_config': env_config}
    )
