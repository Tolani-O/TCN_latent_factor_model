import numpy as np
from scipy.interpolate import BSpline

def generate_bsplines(time, k):
    """
    Generate a set of B-splines with knots c and degree k evaluated at t.
    """
    B = []
    knots = np.concatenate([time, np.repeat(time[-1], k+2)])
    for i in range(len(knots) - (k+2)):
        b = BSpline.basis_element(knots[i:(i+k+2)])
        val = b(time)
        B.append(val)

    B = np.vstack(B)
    # sum rows
    B = B / np.sum(B, axis=0, keepdims=True)

    return B