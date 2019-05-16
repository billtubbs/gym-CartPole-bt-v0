"""
Function to simulate the dynamics of a simple cart-pendulum system.
Copied from MATLAB script provided by Steven L. Brunton as part of
his Control Bootcamp series of YouTube videos.
"""

import math
import numpy as np

# Initialise a dedicated random number generator
rng = np.random.RandomState()

# If you want repeatable results (i.e. not random), use
# rng.seed(seed) before starting each experiment (seed is an
# integer).

def cartpend_dydt(y, m=1, M=5, L=2, g=-10, d=1, u=0, vd=0.01):
    """Simulates the non-linear dynamics of a simple cart-pendulum system.
    These non-linear ordinary differential equations (ODEs) return the
    time-derivative at the current time given the current state of the
    system.

    Args:
        y (np.array): State vector. This should be an array of
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
        vd (float): Variance of random disturbances applied to
            dtheta/dt.

    Returns:
        dy (np.array): The time derivate of the state (dy/dt) as a
            shape (4, ) array.
    """

    # Temporary variables
    Sy = math.sin(y[2])
    Cy = math.cos(y[2])
    mL = m*L
    D = 1/(L*(M + m*(1 - Cy**2)))
    b = mL*y[3]**2*Sy - d*y[1] + u
    dy = np.zeros(4)

    # Non-linear ordinary differential equations describing
    # simple cart-pendulum system dynamics
    dy[0] = y[1]
    dy[1] = D*(-mL*g*Cy*Sy + L*b)
    dy[2] = y[3]
    dy[3] = D*((m + M)*g*Sy - Cy*b) + vd*rng.randn()

    return dy


def run_unit_tests(show=False):
    """Runs a few unit tests to check that cartpend_dydt function
    calculations are accurate.
    """

    # Fixed parameter values
    m = 1
    M = 5
    L = 2
    g = -10
    d = 1
    u = 0
    vd = 0.0  # No random disturbances

    # Run tests
    y_test_values = {
        0: [0, 0, 0, 0],  # Pendulum down position
        1: [0, 0, np.pi, 0],  # Pendulum up position
        2: [0, 0, 0, 0],
        3: [0, 0, np.pi, 0],
        4: [2.260914, 0.026066, 0.484470, -0.026480]
    }

    u_test_values = {
        0: 0.,
        1: 0.,
        2: 1.,
        3: 1.,
        4: -0.59601
    }

    # dy values below calculated with MATLAB script from
    # Steven L. Brunton's Control Bootcamp videos
    expected_results = {
        0: [0., 0., 0., 0.],
        1: [0., -2.44929360e-16, 0., -7.34788079e-16],
        2: [0., 0.2, 0., -0.1],
        3: [0., 0.2, 0. ,0.1],
        4: [0.026066, 0.670896, -0.026480, -2.625542]
        }

    for i, u in u_test_values.items():
        y = np.array(y_test_values[i])
        dy_calculated = cartpend_dydt(y, m=m, M=M, L=L, g=g, d=d, u=u, vd=vd)
        dy_expected = np.array(expected_results[i])
        assert np.isclose(dy_calculated, dy_expected).all(), f"Test {i} failed"
        if show:
            print(f"Test {i} success")

run_unit_tests()