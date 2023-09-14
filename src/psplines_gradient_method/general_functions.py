import numpy as np
import matplotlib.pyplot as plt


def create_precision_matrix(P):
    Omega = np.zeros((P, P))
    # fill the main diagonal with 2s
    np.fill_diagonal(Omega, 2)
    # fill the subdiagonal and superdiagonal with -1s
    np.fill_diagonal(Omega[1:], -1)
    np.fill_diagonal(Omega[:, 1:], -1)
    # set the last element to 1
    Omega[-1, -1] = 1
    return Omega

def create_first_diff_matrix(P):
    D = np.zeros((P-1, P))
    # fill the main diagonal with -1s
    np.fill_diagonal(D, -1)
    # fill the superdiagonal with 1s
    np.fill_diagonal(D[:, 1:], 1)
    return D


def create_second_diff_matrix(P):
    D = np.zeros((P-2, P))
    # fill the main diagonal with 1s
    np.fill_diagonal(D, 1)
    # fill the subdiagonal and superdiagonal with -2s
    np.fill_diagonal(D[:, 2:], 1)
    np.fill_diagonal(D[:, 1:], -2)
    # set the last element to 1
    D[-1, -1] = 1
    return D


def create_masking_matrix(N, M):
    mat = np.zeros((N, N * M), dtype=int)  # Start with a kxkm matrix of zeros
    rows = np.repeat(np.arange(N), M)
    cols = np.arange(N * M).reshape(N, M).ravel()
    mat[rows, cols] = 1
    return mat


def compute_lambda(B_psi, d, G_star, beta):
    K = len(d)
    J = np.ones((K, B_psi.shape[1]))
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G_star @ np.kron(np.eye(K), beta) @ B_psi
    lambda_ = np.exp(diagdJ_plus_GBetaB)
    return lambda_


def plot_spikes(spikes, x_offset=0):
    # Group entries by unique values of s[0]
    unique_s_0 = np.unique(spikes[0])
    grouped_s = []
    for i in unique_s_0:
        indices = np.where(spikes[0] == i)[0]
        values = (spikes[1][indices] - x_offset)/1000
        grouped_s.append((i, values))
    for group in grouped_s:
        plt.scatter(group[1], np.zeros_like(group[1]) + group[0], s=1, c='black')
    plt.show()


def plot_binned(binned):
    # plot binned spikes
    _, ax = plt.subplots()
    ax.imshow(binned)
    ax.invert_yaxis()
    plt.show()


def plot_bsplines(B, time):
    # plot bsplines
    start = 190
    num_basis = 10
    for i in range(num_basis):
        plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
    plt.show()
