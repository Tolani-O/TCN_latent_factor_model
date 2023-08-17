import numpy as np
from scipy.interpolate import BSpline

def generate_bsplines(time, k, dt):
    """
    Generate a set of B-splines with knots c and degree k evaluated at t.
    """
    B = []
    knots = np.concatenate([time, [time[-1]+(i+1)*dt for i in range(k+2)]])
    for i in range(len(knots) - (k+2)):
        b = BSpline.basis_element(knots[i:(i+k+2)], False)
        val = b(time)
        B.append(val)

    B = np.nan_to_num(np.vstack(B))

    return B