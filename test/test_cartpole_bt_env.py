"""
Unit tests for custom cart-pole environments.

To run these tests use this at the command line:
$ python -m unittest test/test_cartpole_bt_env.py

TODO:
- Test with Euler integration option
"""

import os
import unittest
import numpy as np
import gym
import gym_CartPole_BT
from gym_CartPole_BT.systems.cartpend import cartpend_dxdt, cartpend_ss

from numpy.testing import assert_allclose, assert_array_equal


class TestGymCartPoleBT(unittest.TestCase):
    
    def test_cartpole_bt_env(self):
        """Check cartpole_bt_env environments working correctly."""

        env_names = [
            'CartPole-BT-v0',
            'CartPole-BT-dL-v0',
            'CartPole-BT-dH-v0',
            'CartPole-BT-vL-v0',
            'CartPole-BT-vH-v0',
            'CartPole-BT-dL-vL-v0',
            'CartPole-BT-dH-vH-v0',
            'CartPole-BT-p2-v0',
            'CartPole-BT-p2-dL-v0',
            'CartPole-BT-p2-dH-v0',
            'CartPole-BT-p2-vL-v0',
            'CartPole-BT-p2-vH-v0',
            'CartPole-BT-m2-v0',
            'CartPole-BT-m2-dL-v0',
            'CartPole-BT-m2-dH-v0'
        ]

        variance_levels = {None: 0.0, 'low': 0.01, 'high': 0.2}

        for name in env_names:
            env = gym.make(name)
            self.assertEqual(env.length, 2)
            self.assertEqual(env.masspole, 1)
            self.assertEqual(env.masscart, 5)
            self.assertEqual(env.friction, 1)
            self.assertEqual(env.time_step, 0)
            self.assertEqual(env.tau, 0.05)
            self.assertEqual(env.gravity, -10.0)
            self.assertEqual(env.n_steps, 100)
            self.assertEquals(env.variance_levels, variance_levels)
            if '-dL' in name:
                self.assertEqual(env.disturbances, 'low')
            elif '-dH' in name:
                self.assertEqual(env.disturbances, 'high')
            else:
                self.assertIsNone(env.disturbances)
            if '-vL' in name:
                self.assertEqual(env.initial_state_variance, 'low')
            elif '-vH' in name:
                self.assertEqual(env.initial_state_variance, 'high')
            else:
                self.assertIsNone(env.initial_state_variance)
            self.assertIsNone(env.measurement_error)
            self.assertEqual(env.action_space.shape, (1,))
            self.assertEqual(len(env.state_bounds), 4)
            if '-p2' in name:
                assert_array_equal(env.measured_states, ((0, 2)))
            else:
                assert_array_equal(env.measured_states, ((0, 1, 2, 3)))
            if '-m2' in name:
                assert_allclose(env.initial_state, [-1, 0, 3.1415927, 0])
                assert_allclose(env.goal_state, [1, 0, 3.1415927, 0])
            else:
                assert_allclose(env.initial_state, [0, 0, 3.1415927, 0])
                assert_allclose(env.goal_state, [0, 0, 3.1415927, 0])
            measured_states = env.reset()
            assert_array_equal(measured_states, env.state)
            self.assertEqual(measured_states.shape, env.observation_space.shape)
            self.assertEqual(env.initial_state.shape, (4,))

    def test_cartpend(self):
        """Check calculations in cartpend_dydt function."""

        # Fixed parameter values
        m = 1
        M = 5
        L = 2
        g = -10
        d = 1
        u = 0

        # Run tests
        x_test_values = {
            0: [0, 0, 0, 0],  # Pendulum down position
            1: [0, 0, np.pi, 0],  # Pendulum up position
            2: [0, 0, 0, 0],
            3: [0, 0, np.pi, 0],
            4: [2.260914, 0.026066, 0.484470, -0.026480]
        }

        test_values = {
            0: 0.,
            1: 0.,
            2: 1.,
            3: 1.,
            4: -0.59601
        }

        # dy values below calculated with MATLAB script from
        # Steven L. Brunton's Control Bootcamp videos
        expected_results = {
            0: [0., 0., 0., 0.],
            1: [0., -2.44929360e-16, 0., -7.34788079e-16],
            2: [0., 0.2, 0., -0.1],
            3: [0., 0.2, 0. ,0.1],
            4: [0.026066, 0.670896, -0.026480, -2.625542]
            }

        t = 0.0
        for i, u in test_values.items():
            x = np.array(x_test_values[i])
            dx_calculated = cartpend_dxdt(t, x, m=m, M=M, L=L, g=g, d=d, u=u)
            dx_expected = np.array(expected_results[i])
            assert_allclose(dx_calculated, dx_expected, atol=1e-6)

        # K values below calculated with MATLAB script from
        # Steven L. Brunton's Control Bootcamp videos
        test_values = {
            5: 1,  # Pendulum up position
            6: -1  # Pendulum down position
        }
        expected_results = {
            5: (np.array([[0.0,   1.0,   0.0,   0.0],
                        [0.0,  -0.2,   2.0,   0.0],
                        [0.0,   0.0,   0.0,   1.0],
                        [0.0,  -0.1,   6.0,   0.0]]),
                np.array([[ 0.0], [ 0.2], [ 0.0], [ 0.1]])),
            6: (np.array([[0.0,   1.0,   0.0,   0.0],
                        [0.0,  -0.2,   2.0,   0.0],
                        [0.0,   0.0,   0.0,   1.0],
                        [0.0,   0.1,  -6.0,   0.0]]),
                np.array([[ 0.0], [ 0.2], [ 0.0], [-0.1]]))
        }
        for i, s in test_values.items():
            A_calculated, B_calculated = cartpend_ss(m=m, M=M, L=L, g=g, d=d, s=s)
            A_expected, B_expected = expected_results[i]
            assert_allclose(A_calculated, A_expected)
            assert_allclose(B_calculated, B_expected)


if __name__ == '__main__':

    unittest.main()
