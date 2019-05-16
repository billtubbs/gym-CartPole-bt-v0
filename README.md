# gym-CartPole-bt-v0
This is a modified version of the [cart-pole OpenAI Gym environment](https://gym.openai.com/envs/CartPole-v1/) for testing
different controllers and reinforcement learning algorithms.

<IMG SRC="images/screenshot.png" WIDTH=600 ALT="Screenshot">

This version of the classic cart-pole or [cart-and-inverted-pendulum](https://en.wikipedia.org/wiki/Inverted_pendulum#Inverted_pendulum_on_a_cart)
control problem offers more variations on the basic OpenAI
Gym version ('CartPole-v1').

It is based on a MATLAB implementation by [Steven L. Brunton](https://www.me.washington.edu/facultyfinder/steve-brunton)
as part of his excellent [Control Bootcamp](https://youtu.be/qjhAAQexzLg) series of videos on
YouTube.

Features of this version include:
- More challenging control objectives such as stabilizing
  the cart x-position as well as the pendulum angle
- Continuously varying control actions
- Random disturbance to the state [***Not Implemented Yet***]
- Measurement noise [***Not Implemented Yet***]
- Less measurements of state variables [***Not Implemented Yet***]

The goal of building this environment was to test different control 
engineering and reinforcement learning methods on a problem that 
is more challenging than the simple cart-pole environment provided 
by OpenAI but still simple enough to understand and help us learn
about the relative strengths and weaknesses of control/RL 
approaches.


## Installation

To install this environment, first clone or download this repository, then
go to the `gym-CartPole-bt-v0` folder on your computer and run the 
following command in your terminal:

```
pip install -e .
```

This will install the gym environment.  To use the new gym environment in
Python do the following:

```Python
import gym
import gym_CartPole_BT

env = gym.make('CartPole-BT-v0')
```


## Basic usage (without graphics)

```Python
import gym
import gym_CartPole_BT
import numpy as np

# Create and initialize environment
env = gym.make('CartPole-BT-v0')
env.reset()

# Control vector (shape (1, ) in this case)
u = np.zeros(1)

# We will keep track of the cumulative rewards
cum_reward = 0.0

print(f"{'i':>3s}  {'u':>5s} {'reward':>6s} {'cum_reward':>10s}")
print("-"*22)

# Run one episode
done = False
while not done:

    # Retrieve the system state
    x, x_dot, theta, theta_dot = env.state

    # Decide control action (force on cart)
    u[0] = 0.0  # REPLACE THIS WITH YOUR CONTROLLER

    # Run simulation one time-step
    observation, reward, done, info = env.step(u)

    # Process the reward
    cum_reward += reward

    # Print updates
    print(f"{env.time_step:3d}: {u[0]:5.1f} {reward:6.2f} {cum_reward:10.1f}")
```

For demo with graphics animation see [test_run.py](test_run.py)
