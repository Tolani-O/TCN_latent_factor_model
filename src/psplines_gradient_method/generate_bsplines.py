import numpy as np
from scipy.interpolate import BSpline


def generate_bspline_functions(time, degree, deriv=False):
    B = []
    k = degree
    knots = np.concatenate([np.repeat(time[0], k), time, np.repeat(time[-1], k)])
    k_ = k - 1 if deriv else k
    for i in range(len(knots)-(k+1)):
        b = BSpline.basis_element(knots[i:(i + k_ + 2)], False)
        B.append(b)
    return B


def generate_bspline_matrix(bspline_functions, time_matrix):
    K, T = time_matrix.shape
    P = len(bspline_functions)
    B = np.empty((K * P, T))
    for p, bspline_func in enumerate(bspline_functions):
        B[p::P] = bspline_func(time_matrix)
    B = np.nan_to_num(B)
    B[(P-1)::P, -1] = 1
    return B


def bspline_deriv_multipliers(time, degree):
    k = degree
    P = len(time) + 2
    knots = np.concatenate([np.repeat(time[0], k), time, np.repeat(time[-1], k)])
    knots_1 = (1/(knots[k:(P+k)] - knots[0:P]))[:, np.newaxis]
    knots_2 = (1/(knots[(k+1):(P+k+1)] - knots[1:(P+1)]))[:, np.newaxis]
    knots_1[0, 0] = 0
    knots_1[-1, -1] = 0
    knots_2[0, 0] = 0
    knots_2[-1, -1] = 0
    return knots_1, knots_2
