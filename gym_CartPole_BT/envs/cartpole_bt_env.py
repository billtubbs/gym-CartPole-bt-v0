"""
Modified cart-pole environment for use with OpenAI Gym.

This version of the classic cart-pole or cart-and-pendulum
control problem offers more variations on the basic OpenAI
Gym version (CartPole-v1).

It is based on a MATLAB implementation by Steven L. Brunton
as part of his Control Bootcamp series of videos on YouTube.

Features of this version include:
- More challenging control objectives (e.g. to stabilize
  the cart x-position as well as the pendulum angle)
- Continuously varying control actions
- Random disturbance to the state
- Measurement noise
- Reduced set of measured state variables

The goal of building this environment was to test different
control engineering and reinfircement learning methods on
a problem that is more challenging than the simple cart-pole
environment provided by OpenAI but still simple enough to
understand and use to help us learn about the relative
strengths and weaknesses of control/RL approaches.
"""

import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from cartpend import cartpend_dydt

class CartPoleBTEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.gravity = -10.0
        self.masscart = 5.0
        self.masspole = 1.0
        self.length = 2.0
        self.friction = 1.0
        self.max_force = 10.0  # TBC
        self.tau = 0.02   # seconds between state updates
        self.seed()
        self.viewer = None
        self.state = None
        self.n_steps = 10
        self.goal_state = np.array([0.0, 0.0, np.pi, 0.0])

        # Angle and position at which episode fails
        self.theta_threshold_radians = 45*math.pi/360
        self.x_threshold = 2.4

        # Angle limit set to 2*theta_threshold_radians so failing observation is
        # still within bounds
        high = np.array([
            self.x_threshold*2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians*2,
            np.finfo(np.float32).max])

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-self.max_force, self.max_force,
                                       shape=(1,), dtype=np.float32)

        self.time_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_function(y, goal_state):

        return (y[0] - self.goal_state[0])**2 + \
               (angle_normalize(y[2]) - self.goal_state[2])**2

    def step(self, u):

        u = np.clip(u, -self.max_force, self.max_force)[0]

        # Calculate time derivative
        y_dot = cartpend_dydt(
            y=self.state,
            m=self.masspole,
            M=self.masscart,
            L=self.length,
            g=self.gravity,
            d=self.friction,
            u=u,
            vd=0.0  # No disturbances
        )

        # Update state (Euler method)
        self.state += self.tau*y_dot

        reward = self.reward_function(self.state, self.goal_state)

        if self.time_step >= n_steps:
            logger.warn("You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive "
                        "'done = True'")

        self.time_step += 1
        done = True if self.time_step >= n_steps else False

        return self.state, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4, ))
        self.time_step = 0
        return np.array(self.state)

    def render(self, mode='human', close=False):
        raise NotImplementedError

def angle_normalize(x):
    return (((x + np.pi) % (2*np.pi)) - np.pi)