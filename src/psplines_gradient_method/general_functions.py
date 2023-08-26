import numpy as np


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


def compute_lambda(B, d, G, beta):
    J = np.ones((len(d), B.shape[1]))
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + np.dot(np.dot(G, beta), B)
    lambda_ = np.exp(diagdJ_plus_GBetaB)
    return lambda_


def compute_latent_factors(B, beta):
    N = beta @ B
    return N


def compute_numerical_grad(Y, B, d, G, beta, beta_tausq, dt, obj_func, eps=1e-6):
    """Compute numerical gradient of func w.r.t G"""
    G_grad = np.zeros_like(G)
    beta_grad = np.zeros_like(beta)
    d_grad = np.zeros_like(d)
    beta_tausq_grad = np.zeros_like(beta_tausq)
    result = obj_func(Y, B, d, G, beta, beta_tausq, dt)
    lk1 = result["loss"]
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            orig = G[i, j]
            G[i, j] = orig + eps
            result = obj_func(Y, B, d, G, beta, beta_tausq, dt)
            lk2 = result["loss"]
            G[i, j] = orig
            G_grad[i, j] = (lk2 - lk1) / eps

    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            orig = beta[i, j]
            beta[i, j] = orig + eps
            result = obj_func(Y, B, d, G, beta, beta_tausq, dt)
            lk2 = result["loss"]
            beta[i, j] = orig
            beta_grad[i, j] = (lk2 - lk1) / eps

    for i in range(d.shape[0]):
        orig = d[i]
        d[i] = orig + eps
        result = obj_func(Y, B, d, G, beta, beta_tausq, dt)
        lk2 = result["loss"]
        d[i] = orig
        d_grad[i] = (lk2 - lk1) / eps

    for i in range(beta_tausq.shape[0]):
        orig = beta_tausq[i]
        beta_tausq[i] = orig + eps
        result = obj_func(Y, B, d, G, beta, beta_tausq, dt)
        lk2 = result["loss"]
        beta_tausq[i] = orig
        beta_tausq_grad[i] = (lk2 - lk1) / eps

    return -d_grad, -G_grad, -beta_grad, -beta_tausq_grad
