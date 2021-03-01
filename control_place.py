# Design a linear controller using pole-placement method

import numpy as np
import gym
import control
import gym_CartPole_BT
from gym_CartPole_BT.systems.cartpend import cartpend_ss

control.use_numpy_matrix(False)

env = gym.make('CartPole-BT-v0')
env.reset()


m = env.masspole  # Mass of pendulum
M = env.masscart  # Mass of cart
L = env.length  # Length of pendulum
g = env.gravity  # Acceleration due to gravity
d = env.friction  # Damping coefficient for friction between cart and ground

s = 1  # 1 for pendulum up position or -1 for down position

# Generate state space model matrices for linearized system
A, B = cartpend_ss(m, M, L, g, d, s)

# Choose poles of desired closed-loop system
# The poles determine the speed of the controller in 
# manipulating each state
p = [-1, -2, -5, -2.5]

# Compute gain matrix using pole placement
K = control.place(A, B, p)

print(f"Controller gain matrix:\nK = {K.__repr__()}")