#!/usr/bin/env python
# Demonstration of a linear controller using full
# state feedback.

import gym
import gym_CartPole_BT
import numpy as np
import argparse

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-dL-v0',
                    help="gym environment")
parser.add_argument('-s', "--show", help="display output",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
args = parser.parse_args()

# Create and initialize environment
if args.show: print(f"\nInitializing environment '{args.env}'...")
env = gym.make(args.env)
env.reset()

# Control vector (shape (1, ) in this case)
u = np.zeros(1)

# Open graphics window and draw animation
if args.render: env.render()

# We will keep track of the cumulative rewards
cum_reward = 0.0

if args.show:
    print(f"{'k':>3s}  {'u':>5s} {'reward':>6s} {'cum_reward':>10s}")
    print("-"*28)

# Gain matrix (K) for optimal control
# (Calculated using lqr function with Q=np.eye(4), and R=0.0001)
gain = np.array([-100.00,   -197.54,   1491.28,    668.44])

# Run one episode
done = False
while not done:

    # Access the full state
    x, x_dot, theta, theta_dot = env.state

    # Linear quadratic regulator
    # u[t] = -Ky[t]
    u[:] = -np.dot(gain, env.state - env.goal_state)

    # Run simulation one time-step
    observation, reward, done, info = env.step(u)

    # Update the animation
    if args.render: env.render()

    # Process the reward
    cum_reward += reward

    # Print updates
    if args.show:
        print(f"{env.time_step:3d}: {u[0]:5.1f} {reward:6.2f} "
              f"{cum_reward:10.1f}")

input("Press enter to continue...")

# Close animation window
env.close()