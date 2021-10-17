import numpy as np
import control
import gym_CartPole_BT
from gym_CartPole_BT.systems.cartpend import cartpend_ss

A, B = cartpend_ss()

C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
M = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

n = A.shape[0]
ny = C.shape[0]
nm = M.shape[0]
nu = B.shape[1]

# Check system is controllable
assert(np.linalg.matrix_rank(control.ctrb(A, B)) == 4)

tau = 0.05  # sampling rate

# Modal control
I = np.eye(n)

desired_poles = np.array([-1/10, -1/8])

# Desired eigenvalues
eigs = np.exp(-tau / desired_poles)

zero_nu = np.zeros([1, nu])
vpsi1 = np.block([[A - eigs[0] * I, B], [C[0, :], zero_nu], [C[1, :], zero_nu]])
vpsi2 = np.block([[A - eigs[1] * I, B], [C[0, :], zero_nu], [C[1, :], zero_nu]])

vpsi1 = null([A - eigs[1] * I B; C(1,:) zero_nu; C(2,:) zero_nu])
vpsi2 = null([A - eigs[1] * I B; C(1,:) zero_nu; C(3,:) zero_nu])
vpsi3 = null([A - eigs[2] * I B; C(2,:) zero_nu; C(3,:) zero_nu])

# Récupération des vecteur vi et psii
v1=vpsi1(1:n,1)
psi1=vpsi1(n+1:end,1)
v2=vpsi2(1:n,1)
psi2=vpsi2(n+1:end,1)
v3=vpsi3(1:n,1)
psi3=vpsi3(n+1:end,1)

# Formation des matrice Vv et PSI
Vv=[v1 v2 v3]
PSI=[psi1 psi2 psi3]


#%%%%%%%%%% CALCUL DE K %%%%%%%%%% 
K=PSI/(M*Vv)


#%%%%%%%%%% CALCUL DE F %%%%%%%%%% 
F=inv(C*inv(I-A-B*K*M)*B)






# If B and M are full rank, the number of poles that can
# be placed is

nv = max(nu, nm)
assert(np.linalg.matrix_rank(B) == nu)
assert(np.linalg.matrix_rank(M) == nm)

