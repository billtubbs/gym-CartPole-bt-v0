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
    print(f"\n{'k':>3s}  {'u':>5s} {'reward':>6s} {'cum_reward':>10s}")
    print("-"*28)

# Run one episode
done = False
while not done:

    # Retrieve the system state
    x, x_dot, theta, theta_dot = env.state

    # Decide control action (force on cart)
    u[0] = 0.0  # REPLACE THIS WITH YOUR CONTROLLER

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

# Close animation window
env.close()
