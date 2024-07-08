import casadi as cas
import rockit
from rockit import MultipleShooting, FreeTime
import matplotlib.pyplot as plt


# Number of control intervals
N = 30

# Define optimal control problem
ocp = rockit.Ocp(T=FreeTime(2))

# System states
x1 = ocp.state()  # horizontal position of the cart (x)
x2 = ocp.state()  # velocity of the cart (dx/dt)
x3 = ocp.state()  # angle of pendulum from the vertical in radians (theta)
x4 = ocp.state()  # rate of change of theta (dtheta/dt)

# Control action
u = ocp.control()  # force on cart in x-direction

# System parameters
m = 1  # mass of pendulum
M = 5  # mass of cart
L = 2  # length of pendulum
g = -10  # acceleration due to gravity
d = 1  # damping coefficient for friction between cart and ground

# Intermediate expressions
cos_x3 = cas.cos(x3)
sin_x3 = cas.sin(x3)
D = 1 / (L * (M + m * (1 - cos_x3**2)))
b = m * L * x4**2 * sin_x3 - d * x2 + u

# Righthand side of pendulum ODEs
ocp.set_der(x1, x2)
ocp.set_der(x2, D * (-m * L * g * cos_x3 * sin_x3 + L * b))
ocp.set_der(x3, x4)
ocp.set_der(x4, D * ((m + M) * g * sin_x3 - cos_x3 * b))

# Initial condition
ocp.subject_to(ocp.at_t0(x1)==-2.0)
ocp.subject_to(ocp.at_t0(x2)==0.0)
ocp.subject_to(ocp.at_t0(x3)==3.14159)
ocp.subject_to(ocp.at_t0(x4)==0.0)

# Terminal condition
ocp.subject_to(ocp.at_tf(x1)==0.0)
ocp.subject_to(ocp.at_tf(x2)==0.0)
ocp.subject_to(ocp.at_tf(x3)==3.14159)
ocp.subject_to(ocp.at_tf(x4)==0.0)

# Path constraints
ocp.subject_to(-4.8 <= (x1 <= 4.8))  # control action space
ocp.subject_to(2.7236 <= (x3 <= 3.5596))  # stay within 24˚ of vertical
ocp.subject_to(-200 <= (u <= 200))  # control action space

ocp.add_objective(ocp.T)

ocp.solver('ipopt')
ocp.method(MultipleShooting(N=N, intg='rk'))

sol = ocp.solve()

t, u_sol = sol.sample(u, grid='control')
t, x1_sol = sol.sample(x1, grid='control')
t, x2_sol = sol.sample(x2, grid='control')
t, x3_sol = sol.sample(x3, grid='control')
t, x4_sol = sol.sample(x4, grid='control')


plot_info = {
    '$x_1$': x1_sol,
    '$x_2$': x2_sol,
    '$x_3$': x3_sol,
    '$x_4$': x4_sol,
    '$u$': u_sol
}

fig, axes = plt.subplots(len(plot_info), 1, sharex=True, figsize=(7, 8))
for ax, (label, data) in zip(axes, plot_info.items()):
    ax.plot(t, data, '.-')
    ax.set_ylabel(label)
    ax.grid()

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()

breakpoint()
