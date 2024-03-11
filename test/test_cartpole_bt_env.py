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

    def setUp(self):
        self.env_names = [
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
            'CartPole-BT-x2-v0',
            'CartPole-BT-x2-dL-v0',
            'CartPole-BT-x2-dH-v0'
        ]

        self.assertEqual(len(self.env_names), len(set(self.env_names)))

        self.deterministic_envs = [
            'CartPole-BT-v0', 
            'CartPole-BT-p2-v0', 
            'CartPole-BT-x2-v0'
        ]

        # Expected states after two time-steps with default seed
        self.output_test_values = {
            'CartPole-BT-v0': [  0.28277007,  -0.91107196,   3.29043078,  -0.34565255],
            'CartPole-BT-dL-v0': [  0.28276974,  -0.91108167,   3.29040146,  -0.34530669],
            'CartPole-BT-dH-v0': [  0.28276426,  -0.91126573,   3.28984332,  -0.33873528],
            'CartPole-BT-vL-v0': [  0.28648716,  -0.90140116,   3.28294110,  -0.36167920],
            'CartPole-BT-vH-v0': [  0.35633439,  -0.71497697,   3.13958001,  -0.65848947],
            'CartPole-BT-dL-vL-v0': [  0.28648686,  -0.90141082,   3.28291178,  -0.36133304],
            'CartPole-BT-dH-vH-v0': [  0.35632497,  -0.71514016,   3.13899660,  -0.65141183],
            'CartPole-BT-p2-v0': [  0.28277007,   3.29043078],
            'CartPole-BT-p2-dL-v0': [  0.28276974,   3.29040146],
            'CartPole-BT-p2-dH-v0': [  0.28276426,   3.28984332],
            'CartPole-BT-p2-vL-v0': [  0.28648716,   3.28294110],
            'CartPole-BT-p2-vH-v0': [  0.35633439,   3.13958001],
            'CartPole-BT-x2-v0': [ -0.71722996,  -0.91107196,   3.29043078,  -0.34565252],
            'CartPole-BT-x2-dL-v0': [ -0.71723026,  -0.91108167,   3.29040146,  -0.34530655],
            'CartPole-BT-x2-dH-v0': [ -0.71723574,  -0.91126567,   3.28984332,  -0.33873525]
        }

        return super().setUp()

    def sim_test(self, env, inputs, seed=None):
        """Run a test simulation.
        """
        if seed is not None:
            env.seed(seed)
        data = [env.reset()]
        for u in inputs:
            data.append(env.step(u))
        return data

    def test_cartpole_bt_env(self):
        """Check cartpole_bt_env environments working correctly.
        """

        variance_levels = {None: 0.0, 'low': 0.01, 'high': 0.2}

        # Test default environment initialization (no preset random seed)
        for name in self.env_names:
            env = gym.make(name)
            self.assertEqual(env.length, 2)
            self.assertEqual(env.masspole, 1)
            self.assertEqual(env.masscart, 5)
            self.assertEqual(env.friction, 1)
            self.assertEqual(env.time_step, 0)
            self.assertEqual(env.tau, 0.05)
            self.assertEqual(env.gravity, -10.0)
            self.assertEqual(env.n_steps, 100)
            self.assertEqual(env.variance_levels, variance_levels)
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
                assert_array_equal(env.output_matrix, ((1, 0, 0, 0), (0, 0, 1, 0)))
            else:
                assert_array_equal(env.output_matrix, np.eye(4))
            self.assertEqual(env.output_matrix.dtype, np.dtype('float32'))
            if '-x2' in name:
                assert_allclose(env.initial_state, [-1, 0, 3.1415927, 0])
                assert_allclose(env.goal_state, [1, 0, 3.1415927, 0])
            else:
                assert_allclose(env.initial_state, [0, 0, 3.1415927, 0])
                assert_allclose(env.goal_state, [0, 0, 3.1415927, 0])
            self.assertEqual(env.initial_state.dtype, np.dtype('float32'))
            self.assertEqual(env.goal_state.dtype, np.dtype('float32'))
            initial_output = env.reset()
            self.assertEqual(initial_output.dtype, np.dtype('float32'))
            self.assertEqual(initial_output.shape, env.observation_space.shape)
            self.assertEqual(env.state.shape, (4,))
            self.assertEqual(env.state.dtype, np.dtype('float32'))
            if '-vL' in name or '-vH' in name:
                self.assertFalse(np.array_equal(initial_output, 
                                                env.output(env.initial_state)))
            else:
                self.assertTrue(np.array_equal(initial_output, 
                                               env.output(env.initial_state)))

            # Simulate one time step
            u = np.array([1.0])
            output_1, reward, done, info = env.step(u)
            self.assertEqual(env.time_step, 1)
            self.assertEqual(output_1.dtype, np.dtype('float32'))
            self.assertFalse(done)
            if '-p2' in name:
                self.assertEqual(output_1.shape, (2,))
            else:
                self.assertEqual(output_1.shape, (4,))

            # Simulate 2nd time step
            u = np.array([-250.0])  # Exceeds the limit
            output_2, reward, done, info = env.step(u)
            self.assertEqual(env.time_step, 2)
            self.assertFalse(np.isclose(output_1, output_2).all())

            # Check environment reset
            output_3 = env.reset()
            self.assertEqual(env.time_step, 0)
            if '-vL' in name or '-vH' in name:
                self.assertFalse(np.isclose(output_3, initial_output).all())
            else:
                self.assertTrue(
                    np.array_equal(output_3, env.output(env.initial_state))
                )
            u = np.array([1.0])
            output_1r, reward, done, info = env.step(u)

            # Check states of deterministic environments after 2 time-steps
            if name in self.deterministic_envs:
                # Check they were the same after reset
                self.assertTrue(np.array_equal(output_1r, output_1))

            #else:  # TODO: understand this
            #    if np.array_equal(output_1r, output_1):
            #        breakpoint()
            #    # self.assertFalse()

        # Check stochastic environments are repeatable
        stochastic_envs = [name for name in self.env_names 
                           if name not in self.deterministic_envs]

        # Run longer simulations with preset random seeds
        inputs = [[100.0], [100.0], [10], [-100], [-200]]  # for test sim
        for name in self.env_names:
            seeds = [1, 10]
            envs = [gym.make(name) for seed in seeds]
            data = {env: {} for env in envs}
            for env, seed in zip(envs, seeds):
                # Run simulation
                data[env]['Test 1'] = self.sim_test(env, inputs, seed)
                # Check output changed
                y1 = data[env]['Test 1'][0]  # Initial output (y0)
                y2 = data[env]['Test 1'][1][0]  # Output after 1st timestep
                self.assertFalse(np.isclose(y1, y2).all())

                # Reset and repeat
                data[env]['Test 2'] = self.sim_test(env, inputs, seed)
                # Each data item contains: (output, reward, done, info)

                # Check outputs are the same for both simulations
                y2 = data[env]['Test 2'][0]
                self.assertTrue(np.array_equal(y1, y2))
                y1 = data[env]['Test 1'][-1][0]  # Final output
                y2 = data[env]['Test 2'][-1][0]
                self.assertTrue(np.array_equal(y1, y2))

                # Check rewards are same
                r1 = data[env]['Test 1'][1][1]  # First reward
                r2 = data[env]['Test 2'][1][1]
                assert_allclose(r1, r2)
                r1 = data[env]['Test 1'][-1][1]  # Final reward
                r2 = data[env]['Test 2'][-1][1]
                assert_allclose(r1, r2)

            # Check final states in first test case
            yN = data[envs[0]]['Test 1'][-1][0]
            #Use the following to re-generate the test values in SetUp above
            #array_as_str = ', '.join(f"{x:12.8f}" for x in yN)
            #print(f"{name.__repr__()}: [{array_as_str}],")
            expected_output = self.output_test_values[name]
            assert_allclose(yN, expected_output)

            # Check output of seeded stochastic environments is different
                    # TODO: is this correct?
            if name in stochastic_envs:
                y1 = data[envs[0]]['Test 1'][2][0]  # Output after 2nd step
                y2 = data[envs[1]]['Test 1'][2][0]
                self.assertFalse(np.array_equal(y1, y2))

    def test_cartpend(self):
        """Check calculations in cartpend_dydt function.
        """

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
