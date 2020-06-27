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
from functools import partial
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import solve_ivp
from gym_CartPole_BT.systems.cartpend import cartpend_dydt

class CartPoleBTEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a track. The goal is to move the cart and the pole to a goal position
        and angle and stabilize it.

    Source:
        This environment corresponds to the version of the cart-pendulum problem
        described by Steven L. Brunton in his Control Bootcamp series of YouTube
        videos.

    Observations:
        Type: Box(4)
        Num	Observation                Min           Max
        0	Cart Position             -Inf           Inf
        1	Cart Velocity             -Inf           Inf
        2	Pole Angle (radians)      -Inf           Inf
        3	Pole Angular Velocity     -Inf           Inf

    Actions:
        Type: Box(1)
        Num	Action                     Min           Max
        0	Force on Cart             -200           200

    Reward:
        The reward is calculated each time step and is a negative cost.
        The cost function is the sum of the squared differences between
          (i) the cart x-position and the goal x-position
         (ii) the pole angle and the goal angle

    Starting State:
        Each episode, the system starts in a random state.

    Episode Termination:
        Episode ends after 100 timesteps.

    Solved Requirements:
        To be determined by comparison with the ideal controller.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, description="Cart-pendulum system",
                 goal_state=(0.0, 0.0, np.pi, 0.0),
                 disturbances=None,
                 initial_state='goal',
                 initial_state_variance=None,
                 measurement_error=None,  # Not implemented yet
                 hidden_states=None,  # Not implemented yet
                 n_steps=100
                 ):

        self.description = description

        # Physical attributes of system
        self.gravity = -10.0
        self.masscart = 5.0
        self.masspole = 1.0
        self.length = 2.0
        self.friction = 1.0
        self.max_force = 200.0

        # Set initial state and goal state
        self.goal_state = np.array(goal_state, dtype=np.float32)
        if initial_state == 'goal':
            self.initial_state = self.goal_state.copy()
        else:
            self.initial_state = np.array(initial_state, dtype=np.float32)

        # Other features
        self.disturbances = disturbances
        self.initial_state_variance = initial_state_variance
        self.measurement_error = measurement_error
        if hidden_states is None:
            self.output_matrix = np.eye(4)  # Not implemented yet
        self.variance_levels = {
            None: 0.0,
            'low': 0.01,
            'high': 0.2
        }

        # Details of simulation
        self.tau = 0.05   # seconds between state updates
        self.n_steps = n_steps
        self.time_step = 0
        self.kinematics_integrator = 'RK45'

        # Maximum and minimum angle and cart position
        # TODO: If episode doesn't terminate limits should be inf.
        self.theta_threshold_radians = np.finfo(np.float32).max
        self.x_threshold = np.finfo(np.float32).max

        # Thresholds for observation bounds
        high = np.array([
            self.x_threshold,
            np.finfo(np.float32).max,
            self.theta_threshold_radians,
            np.finfo(np.float32).max],
            dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(np.float32(-self.max_force), 
                                       np.float32(self.max_force),
                                       shape=(1,), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cost_function(self, state, goal_state):
        """Evaluates the cost based on the current state y and
        the goal state.
        """

        return ((state[0] - self.goal_state[0])**2 +
                (angle_normalize(state[2]) - self.goal_state[2])**2)

    def step(self, u):

        u = np.clip(u, -self.max_force, self.max_force)[0]
        y = self.state
        t = self.time_step * self.tau

        if self.kinematics_integrator == 'Euler':
            # Calculate time derivative
            y_dot = cartpend_dydt(t, y,
                                  m=self.masspole,
                                  M=self.masscart,
                                  L=self.length,
                                  g=self.gravity,
                                  d=self.friction,
                                  u=u)

            # Simple state update (Euler method)
            self.state += self.tau * y_dot

        else:
            # Create a partial function for use by solver
            f = partial(cartpend_dydt,
                        m=self.masspole,
                        M=self.masscart,
                        L=self.length,
                        g=self.gravity,
                        d=self.friction,
                        u=u)

            # Integrate using numerical solver
            tf = t + self.tau
            sol = solve_ivp(f, t_span=[t, tf], y0=self.state,
                            method=self.kinematics_integrator, 
                            t_eval=[tf])
            self.state = sol.y.reshape(-1)

        # Add disturbance only to pendulum angular velocity (theta_dot)
        if self.disturbances is not None:
            v = self.variance_levels[self.disturbances]
            self.state[3] += 0.05 * self.np_random.normal(scale=v)

        reward = -self.cost_function(self.state, self.goal_state)

        if self.time_step >= self.n_steps:
            logger.warn("You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive "
                        "'done = True'")

        self.time_step += 1
        done = True if self.time_step >= self.n_steps else False

        return self.state, reward, done, {}

    def reset(self):

        self.state = self.initial_state.copy()
        assert self.state.shape[0] == 4

        # Add random variance
        v = self.variance_levels[self.initial_state_variance]
        self.state += self.np_random.normal(scale=v, size=(4, ))
        self.time_step = 0
        return self.state

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 0.5
        scale = screen_width/world_width
        carty = 160 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (0.5*self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:

            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = (-cartwidth/2, cartwidth/2, cartheight/2,
                          -cartheight/2)
            axleoffset = cartheight/4.0

            # Draw cart
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (-polewidth/2, polewidth/2, polelen - polewidth/2,
                          -polewidth/2)

            # Draw pole
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self._pole_geom = pole

            # Draw axle
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            # Draw track
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # Draw goal line
            x = screen_width/2.0 + self.goal_state[0] * scale
            self.goal_line = rendering.Line((x, carty),
                                            (x, carty + polelen + 25))
            self.goal_line.set_color(0, 0, 0)
            self.viewer.add_geom(self.goal_line)

            # Draw initial state position
            if self.initial_state[0] != self.goal_state[0]:
                x = screen_width/2.0 + self.initial_state[0] * scale
                self.init_line = rendering.Line((x, carty),
                                                (x, carty + polelen + 25))
                self.init_line.set_color(0, 0, 0)
                self.viewer.add_geom(self.init_line)

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (-polewidth/2, polewidth/2, polelen - polewidth/2,
                      -polewidth/2)
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0]*scale + screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2] + np.pi)  # -x[2]

        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(theta):
    return theta % (2*np.pi)
