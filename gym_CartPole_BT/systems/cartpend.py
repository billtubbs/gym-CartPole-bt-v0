"""
Functions to simulate the dynamics of a simple cart-pendulum system.
Copied from MATLAB script provided by Steven L. Brunton as part of his
Control Bootcamp series of YouTube videos.
"""

import math
import numpy as np


def cartpend_dxdt(t, x, m=1, M=5, L=2, g=-10, d=1, u=0):
    """Simulates the non-linear dynamics of a simple cart-pendulum system.
    These non-linear ordinary differential equations (ODEs) return the
    time-derivative at time t given the current state of the system.

    Args:
        t (float): Time variable - not used here but included for
            compatibility with solvers like scipy.integrate.solve_ivp.
        x (np.array): State vector. This should be an array of
            shape (4, ) containing the current state of the system.
            y[0] is the x-position of the cart, y[1] is the velocity
            of the cart (dx/dt), y[2] is the angle of the pendulum
            (theta) from the vertical in radians, and y[3] is the
            rate of change of theta (dtheta/dt).
        m (float): Mass of pendulum.
        M (float): Mass of cart.
        L (float): Length of pendulum.
        g (float): Acceleration due to gravity.
        d (float): Damping coefficient for friction between cart and
            ground.
        u (float): Force on cart in x-direction.

    Returns:
        dx (np.array): The time derivate of the state (dx/dt) as an
            array of shape (4,).
    """

    # Temporary variables
    sin_x = math.sin(x[2])
    cos_x = math.cos(x[2])
    mL = m * L
    D = 1 / (L * (M + m * (1 - cos_x**2)))
    b = mL * x[3]**2 * sin_x - d * x[1] + u
    dx = np.zeros(4)

    # Non-linear ordinary differential equations describing
    # simple cart-pendulum system dynamics
    dx[0] = x[1]
    dx[1] = D * (-mL * g * cos_x * sin_x + L * b)
    dx[2] = x[3]
    dx[3] = D * ((m + M) * g * sin_x - cos_x * b)

    return dx


def cartpend_ss(m=1, M=5, L=2, g=-10, d=1, s=1):
    """Calculates the linearized approximation of the cart-pendulum
    system dynamics at either the vertical-up position (s=1) or
    vertical-down position (s=-1).

    Returns two arrays, A, B which are the system and input matrices
    in the state-space system of differential equations:

        x_dot = Ax + Bu

    where x is the state vector, u is the control vector and x_dot
    is the time derivative (dx/dt).

    Args:
        m (float): Mass of pendulum.
        M (float): Mass of cart.
        L (float): Length of pendulum.
        g (float): Acceleration due to gravity.
        d (float): Damping coefficient for friction between cart and
            ground.
        s (int): 1 for pendulum up position or -1 for down.

    Returns:
        dy (np.array): The time derivate of the state (dy/dt) as a
            shape (4, ) array.
    """

    A = np.array([
        [       0.,         1.,               0.,       0.],
        [       0,        -d/M,           -m*g/M,       0.],
        [       0.,         0.,               0.,       1.],
        [       0., -s*d/(M*L), -s*(m+M)*g/(M*L),       0.]
    ])

    B = np.array([
        [         0.],
        [       1./M],
        [         0.],
        [ s*1./(M*L)]
    ])

    return A, B


# See unit-tests in test/test_cartpole_bt_env.py