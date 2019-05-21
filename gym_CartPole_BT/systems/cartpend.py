"""
Functions to simulate the dynamics of a simple cart-pendulum system.
Copied from MATLAB script provided by Steven L. Brunton as part of his
Control Bootcamp series of YouTube videos.
"""

import math
import numpy as np


def cartpend_dydt(t, y, m=1, M=5, L=2, g=-10, d=1, u=0):
    """Simulates the non-linear dynamics of a simple cart-pendulum system.
    These non-linear ordinary differential equations (ODEs) return the
    time-derivative at the current time given the current state of the
    system.

    Args:
        t (float): Time variable - not used here but included for
            compatibility with solvers like scipy.integrate.solve_ivp.
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
    dy[3] = D*((m + M)*g*Sy - Cy*b)

    return dy

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

    # Run tests
    y_test_values = {
        0: [0, 0, 0, 0],  # Pendulum down position
        1: [0, 0, np.pi, 0],  # Pendulum up position
        2: [0, 0, 0, 0],
        3: [0, 0, np.pi, 0],
        4: [2.260914, 0.026066, 0.484470, -0.026480]
    }

    test_values = {
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

    t = 0.0
    for i, u in test_values.items():
        y = np.array(y_test_values[i])
        dy_calculated = cartpend_dydt(t, y, m=m, M=M, L=L, g=g, d=d, u=u)
        dy_expected = np.array(expected_results[i])
        assert np.isclose(dy_calculated, dy_expected).all(), f"Test {i} failed"
        if show:
            print(f"Test {i} success")

    # K values below calculated with MATLAB script from
    # Steven L. Brunton's Control Bootcamp videos
    test_values = {
        5: 1,  # Pendulum up position
        6: -1  # Pendulum down position
    }
    expected_results = {
        5: (np.array([[0.0,   1.0,   0.0,   0.0],
                      [0.0,  -0.2,   2.0,   0.0],
                      [0.0,   0.0,   0.0,   1.0],
                      [0.0,  -0.1,   6.0,   0.0]]),
            np.array([[ 0.0], [ 0.2], [ 0.0], [ 0.1]])),
        6: (np.array([[0.0,   1.0,   0.0,   0.0],
                      [0.0,  -0.2,   2.0,   0.0],
                      [0.0,   0.0,   0.0,   1.0],
                      [0.0,   0.1,  -6.0,   0.0]]),
            np.array([[ 0.0], [ 0.2], [ 0.0], [-0.1]]))
    }
    for i, s in test_values.items():
        A_calculated, B_calculated = cartpend_ss(m=m, M=M, L=L, g=g, d=d, s=s)
        A_expected, B_expected = expected_results[i]
        assert np.isclose(A_calculated, A_expected).all(), f"Test {i} failed"
        assert np.isclose(B_calculated, B_expected).all(), f"Test {i} failed"
        if show:
            print(f"Test {i} success")

# For now, let's run unit-tests every time this module is imported
run_unit_tests()