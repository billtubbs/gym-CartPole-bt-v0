"""
Unit tests for custom cart-pole environments.

To run these tests use this at the command line:
$ python -m unittest test/test_cartpole_bt_env.py

TODO:
- Add tests for system functions
"""

import os
import unittest
import numpy as np
import gym
import gym_CartPole_BT

from numpy.testing import assert_allclose, assert_array_equal


class TestGymCartPoleBT(unittest.TestCase):
    
    def test_cartpole_bt_env(self):

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


if __name__ == '__main__':

    unittest.main()
