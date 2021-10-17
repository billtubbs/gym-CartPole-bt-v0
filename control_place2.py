import numpy as np
import gym_CartPole_BT
from gym_CartPole_BT.systems.cartpend import cartpend_ss

A, B = cartpend_ss()

C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
M = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

n = A.shape[0]
ny = C.shape[0]
nm = M.shape[0]
nu = B.shape[1]

# If B and M are full rank, the number of poles that can
# be placed is
nv = max(nu, nm)

assert(np.linalg.matrix_rank(B) == nu)
# Problem. M is not full rank

desired_poles = [-1/10 -1/8 ]