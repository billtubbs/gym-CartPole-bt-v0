import numpy as np
import gym
import gym_CartPole_BT
import argparse
import time


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Speed test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-dL-v0',
                    help="gym environment")
parser.add_argument('-s', "--show", help="display output",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('-n', "--n_episodes", type=int, default=100,
                    help="number of episodes to run")
args = parser.parse_args()


class random_policy:
    
    def __init__(self, env):
        self.env = env
    
    def predict(self, state):
        return env.action_space.sample()


def run_episode(env, policy, render=True, show=True):

    obs = env.reset()

    if render:
        # Open graphics window and draw animation
        env.render()

    if show:
        print(f"{'k':>3s}  {'u':>5s} {'reward':>6s} "
               "{'cum_reward':>10s}")
        print("-"*28)

    # Keep track of the cumulative rewards
    cum_reward = 0.0

    # Run one episode
    done = False
    while not done:

        # Determine control input
        u = policy.predict(obs)

        # Run simulation one time-step
        obs, reward, done, info = env.step(u)

        if render:
            # Update the animation
            env.render()

        # Process the reward
        cum_reward += reward

        if show:
            # Print updates
            print(f"{env.time_step:3d}: {u[0]:5.1f} "
                  f"{reward:6.2f} {cum_reward:10.1f}")

    return cum_reward


def run_episodes(env, policy, n_edisodes=100, render=True, show=True):
    results = []
    for i in range(n_episodes):
        cum_reward = run_episode(env, policy, render=args.render, 
                                show=args.verbose)
        results.append(cum_reward)
    return results


# Create and initialize environment
if args.show: print(f"\nInitializing environment '{args.env}'...")
env = gym.make(args.env)
env.reset()

# Initialize policy
if args.show: print(f"\nInitializing random policy...")
policy = random_policy(env)

# Run repeated simulations (with animation if args.render is True)
n_episodes = args.n_episodes
if args.show:
    print(f"\nRunning policy for {n_episodes} episodes...")

t0 = time.perf_counter()
run_episodes(env, policy, n_edisodes=n_episodes)
t1 = time.perf_counter()

if args.show:
    print(f"\nSimulations complete.")

print(f"Duration: {t1 - t0:.6f} seconds")

# Close animation window
env.close()
