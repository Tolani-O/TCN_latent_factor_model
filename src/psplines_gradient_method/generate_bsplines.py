import numpy as np
from scipy.interpolate import BSpline


def generate_bspline_matrix(time, degree):
    B = []
    k = degree
    knots = np.concatenate([np.repeat(time[0], k), time, np.repeat(time[-1], k)])
    for i in range(len(knots)-(k+1)):
        b = BSpline.basis_element(knots[i:(i + k + 2)], False)
        val = b(time)
        B.append(val)

    B = np.nan_to_num(np.vstack(B))  # already adds up to 1
    # set the last element to 1
    B[-1, -1] = 1

    return B
