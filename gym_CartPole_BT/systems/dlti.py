"""
Classes to simulate discrete-time, linear, time-invariant, systems.
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.signal import dlti


class DynamicSystemARMA():
    """Auto-regressive moving average model.
    Defined by the following difference equation:

        y(k) = [ C(q^-1) / A(q^-1) ] * e(k)

    where:
        A(q^-1) = 1 + a1 * q^-1 + a2 * q^-2 + ...
        C(q^-1) = 1 + c1 * q^-1 + c2 * q^-2 + ...
        [a1, a2, ...] and [c1, c2, ...] are constants.
        q^-1 is the backward shift operator.
    """

    def __init__(self, c, a, dtype=np.float32):
        self.c = np.array(c, dtype=dtype)
        self.a = np.array(a, dtype=dtype)
        if len(c) > len(a):
            raise ValueError("Improper transfer function. `len(c) > len(a)`.")
        self.y_past = np.zeros_like(self.a, dtype=dtype)
        self.e_past = np.zeros_like(self.c, dtype=dtype)

    def step(self, ek):
        yk = -np.sum(self.a * self.y_past) + np.sum(self.c * self.e_past) + ek
        if self.a.shape[0] > 0:
            self.y_past[1:] = self.y_past[:-1]
            self.y_past[0] = yk
        if self.c.shape[0] > 0:
            self.e_past[1:] = self.e_past[:-1]
            self.e_past[0] = ek
        return yk

# Test simple integrator: y(k) = y(k-1) + e(k)
c = []
a = [-1]
sys = DynamicSystemARMA(c, a)
assert_array_equal(sys.c, c)
assert sys.c.dtype == np.float32
assert_array_equal(sys.a, a)
assert sys.a.dtype == np.float32
assert_array_equal(sys.y_past, [0])
assert sys.y_past.dtype == np.float32
assert_array_equal(sys.e_past, [])
assert sys.e_past.dtype == np.float32
e_in = [1, 2, 1, 3]
y_out = [sys.step(ek) for ek in e_in]
assert_array_equal(y_out, np.cumsum(e_in))

# Tests
c = [2]
a = [-1]
sys = DynamicSystemARMA(c, a)

# k = 0
e0 = 0.111
y0 = sys.step(e0)
assert y0 == e0

# k = 1
e1 = 0.211
y1 = sys.step(e1)
assert np.isclose(y1, -(-1 * y0) + 2 * e0 + e1)

# k = 2
e2 = -0.311
y2 = sys.step(e2)
assert np.isclose(y2, -(-1 * y1) + 2 * e1 + e2)

# k = 3
e3 = 0.111
y3 = sys.step(e3)
assert np.isclose(y3, -(-1 * y2) + 2 * e2 + e3)

# Compare to Scipy implementation of discrete-time transfer function
a1 = -1.
c1 = 2.
sys = dlti([1, c1], [1, a1])
e_in = [e0, e1, e2, e3]
t = np.arange(4)
t_out, y_out = sys.output(e_in, t)
assert_array_equal(t, t_out)
assert_array_equal(y_out, [
    [0.111], 
    [0.544], 
    [0.655], 
    [0.14400000000000002]
])
assert_array_almost_equal(y_out.reshape(-1), [y0, y1, y2, y3])

# More complex simulation
nt = 50
c = [0.2962, 0.1204, -0.0542]
nc = len(c)
a = [0.8502, 0.1023, 0.0237, 0.0032]
na = len(a)
e_in = np.random.randn(nt)
t = np.arange(nt)

sys = dlti(np.concatenate([[1], c]), np.concatenate([[1], a]))
t_out, y_out = sys.output(e_in, t)

sys = DynamicSystemARMA(c, a)
y_out2 = [sys.step(ek) for ek in e_in]
assert_array_almost_equal(y_out.reshape(-1)[na-nc:], y_out2[:-na+nc])
