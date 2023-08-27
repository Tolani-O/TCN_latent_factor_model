import numpy as np
from scipy.interpolate import BSpline


def generate_bsplines(time, degree):
    """
    Generate a set of B-splines with knots c and degree k evaluated at t.
    """
    B = []
    k = degree
    knots = np.concatenate([np.repeat(time[0], k), time, np.repeat(time[-1], k)])
    for i in range(len(knots) - (k+2)):
        b = BSpline.basis_element(knots[i:(i + k + 2)], False)
        val = b(time)
        B.append(val)

    B = np.nan_to_num(np.vstack(B))  # already adds up to 1
    # set the last element to 1
    B[-1, -1] = 1

    return B
