import numpy as np
import pandas as pd
import gym
import gym_CartPole_BT
from control_baselines import LQR
import argparse

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-dL-v0',
                    help="gym environment")
parser.add_argument('-s', "--show", help="display output",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()

def run_episode(env, model, render=True, show=True):

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
        u, _ = model.predict(obs)

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


# Create and initialize environment
if args.show: print(f"\nInitializing environment '{args.env}'...")
env = gym.make(args.env)
env.reset()

model = LQR(None, env)

# Use random search to find the best linear controller:
# u[t] = -Ky[t]

search_size = 1000.0

search_parameters = {
    'CartPole-BT-v0': (1, 1),
    'CartPole-BT-dL-v0': (100, 3),
    'CartPole-BT-vL-v0': (100, 3),
    'CartPole-BT-dL-vL-v0': (100, 3),
    'CartPole-BT-dH-v0': (500, 5),
    'CartPole-BT-vH-v0': (1000, 5)
}

try:
    n_iter, top_n = search_parameters[env_name]
except:
    n_iter, top_n = (100, 3)

# Start random search over full search area
if args.show:
    print(f"\nStarting random search for {n_iter} episodes...")
results = []
for i in range(n_iter):
    gain = (np.random.random(size=(1, 4)) - 0.5)*search_size
    model = LQR(None, env, gain)
    cum_reward = run_episode(env, model, render=False, show=False)
    results.append((cum_reward, gain))

top_results = pd.DataFrame(results, columns=['cum_reward', 'gain'])
top_results = top_results.sort_values(by='cum_reward', ascending=False)
top_results = top_results.reset_index(drop=True).head(top_n)
if args.show:
    print(f"Top {top_n} results after {n_iter} episodes:\n"
          f"{top_results[['cum_reward']].round(2)}")

best_gain = top_results.loc[0, 'gain']
std_gain = np.vstack(top_results['gain'].values).std(axis=0)

df = pd.DataFrame({'Best': best_gain.ravel(), 'Std.': std_gain})
if args.show:
    print(f"Best gain values and std. dev:\n{df.round(3)}")
    print(f"\nStarting targetted search for {n_iter} episodes...")

# Now search within reduced area
for i in range(n_iter):
    gain = np.random.normal(best_gain, std_gain)
    model = LQR(None, env, gain)
    cum_reward = run_episode(env, model, render=args.render, show=False)
    #print(f"{i}: Cum reward: {cum_reward}")
    results.append((cum_reward, gain))

top_results = pd.DataFrame(results, columns=['cum_reward', 'gain'])
top_results = top_results.sort_values(by='cum_reward', ascending=False)
top_results = top_results.reset_index(drop=True).head(top_n)
if args.show:
    print(f"Top {top_n} results after {n_iter} episodes:\n"
          f"{top_results[['cum_reward']].round(2)}")
    print(f"\nStarting robustness checks on top {top_n} results...")

results = []
# Do a robustness check on top results:
for gain in top_results['gain']:
    model = LQR(None, env, gain)
    # Average over 3 runs
    mean_reward = np.mean([
        run_episode(env, model, render=args.render, show=False)
        for _ in range(3)
    ])
    results.append(mean_reward)

best = np.argmax(results)
best_reward, best_gain = top_results.loc[best]

if args.show:
    print(f"\nBest result (#{best}):")
    print(f" Reward: {round(best_reward, 2)}")
    print(f" Gain: {best_gain.round(3)}")

    input("\nPress enter to start simulation...")

model.gain = best_gain

# Run repeated simulations with animation
if args.show:
    while True:
        cum_reward = run_episode(env, model, render=args.render,
                                 show=args.verbose)
        print(f"Reward: {round(cum_reward, 2)}")
        s = input("Press enter to run again, 'q' to quit: ")
        if s.lower() == 'q':
            break

# Close animation window
env.close()
