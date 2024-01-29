"""
Functions to simulate the dynamics of a simple cart-pendulum system.
Based on the MATLAB script provided by Steven L. Brunton as part of his
Control Bootcamp series of YouTube videos.
"""

import numpy as np


def cartpend_dxdt(t, x, m=1, M=5, L=2, g=-10, d=1, u=0):
    """Simulates the dynamics of a simple cart-pendulum system.
    
    Returns the right-hand side of the following non-linear ordinary 
    differential equation (ODE):
    
    dx/dt(t) = f(x(t))

    which defines the time-derivative of the state variables dx/dt(t) at 
    time t given the current state of the system, x(t).

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
        dxdt (np.array): The time derivate of the state (dx/dt) as an
            array of shape (4,).
    """

    # Temporary variables
    sin_x = np.sin(x[2])
    cos_x = np.cos(x[2])
    mL = m * L
    D = 1 / (L * (M + m * (1 - cos_x**2)))
    b = mL * x[3]**2 * sin_x - d * x[1] + u

    # Non-linear ordinary differential equations describing
    # simple cart-pendulum system dynamics
    dxdt = np.array([
        x[1],
        D * (-mL * g * cos_x * sin_x + L * b),
        x[3],
        D * ((m + M) * g * sin_x - cos_x * b)
    ], dtype=np.float32)

    return dxdt


def cartpend_ss(m=1, M=5, L=2, g=-10, d=1, s=1):
    """Calculates the state-space model matrices for the linearized 
    approximation of the cart-pendulum system at either the 
    vertical-up position (s=1) or vertical-down position (s=-1).

    Returns two arrays, A, the state transition matrix, and B, 
    the input matrix, of the state-space system:

        dx/dt(t) = A.x(t) + B.u(t)

    where x(t) is the state vector, u(t) is a control input  
    vector and dx/dt(t) is the time derivative of x(t) at time t.

    Args:
        m (float): Mass of pendulum.
        M (float): Mass of cart.
        L (float): Length of pendulum.
        g (float): Acceleration due to gravity.
        d (float): Damping coefficient for friction between cart 
            and ground.
        s (int): 1 for pendulum up position or -1 for down.

    Returns:
        A, B (np.arrays): The A, B matrices of the state-space
            model of the linearized system.
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