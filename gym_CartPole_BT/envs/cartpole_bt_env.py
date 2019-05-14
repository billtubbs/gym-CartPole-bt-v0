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

class CartPoleBTEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.gravity = -10.0
        self.masscart = 5.0
        self.masspole = 1.0
        self.length = 2.0
        self.friction = 1.0
        self.force_max = 10.0  # TBC
        self.tau = 0.02   # seconds between state updates
        self.seed()
        self.viewer = None
        self.state = None

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
        self.action_space = spaces.Box(-self.force_max, self.force_max,
                                       shape=(1,), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human', close=False):
        raise NotImplementedError
