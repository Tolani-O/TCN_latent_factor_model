import numpy as np
from numpy.linalg import det


def log_prob(Y, B, d, G, beta, beta_tausq, dt):
    L, P = beta.shape
    J = np.ones_like(Y)
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G @ beta @ B
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    Omega = create_precision_matrix(P)
    BetaOmegaBeta = beta @ Omega @ beta.T

    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    log_prior_beta = 0.5 * (L * np.log((2 * np.pi) ** (-P) * det(Omega)) +
                            P * np.sum(np.log(beta_tausq)) -
                            beta_tausq.T @ np.diag(BetaOmegaBeta))
    loss = log_likelihood + log_prior_beta

    # Manual gradients
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T

    dLogL_dd = np.sum(y_minus_lambdadt, axis=1)
    dlogL_dG = y_minus_lambdadt_times_B @ beta.T
    dlogL_dbeta = (G.T @ y_minus_lambdadt_times_B -
                   beta_tausq[:, np.newaxis] * beta @ Omega)
    dlogL_beta_tausq = 0.5 * (P * beta_tausq ** (-1) - np.diag(BetaOmegaBeta))

    result = {
        "dLogL_dd": -dLogL_dd,
        "dlogL_dG": -dlogL_dG,
        "dlogL_dbeta": -dlogL_dbeta,
        "dlogL_beta_tausq": -dlogL_beta_tausq,
        "loss": loss,
        "log_likelihood": log_likelihood,
        "penalty": log_prior_beta
    }

    return result


def log_obj(Y, B, d, G, beta, beta_tausq, dt):
    L, P = beta.shape
    J = np.ones_like(Y)
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G @ beta @ B
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    Omega = create_first_diff_matrix(P)
    Omega = Omega.T @ Omega
    BetaOmegaBeta = beta @ Omega @ beta.T

    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    penalty = - beta_tausq.T @ np.diag(BetaOmegaBeta)
    loss = log_likelihood + penalty

    # Manual gradients
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T

    dLogL_dd = np.sum(y_minus_lambdadt, axis=1)
    dlogL_dG = y_minus_lambdadt_times_B @ beta.T
    dlogL_dbeta = (G.T @ y_minus_lambdadt_times_B -
                   2 * beta_tausq[:, np.newaxis] * beta @ Omega)

    result = {
        "dLogL_dd": -dLogL_dd,
        "dlogL_dG": -dlogL_dG,
        "dlogL_dbeta": -dlogL_dbeta,
        "dlogL_beta_tausq": 0,
        "loss": loss,
        "log_likelihood": log_likelihood,
        "penalty": penalty
    }

    return result


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
