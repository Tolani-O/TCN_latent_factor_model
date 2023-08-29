import numpy as np
from numpy.linalg import det

from src.psplines_gradient_method.general_functions import create_first_diff_matrix, create_precision_matrix


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


def log_obj(Y, B, d, G, beta, Omega, beta_tausq, G_eta, G_smooth, smooth, dt):
    J = np.ones_like(Y)
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G @ beta @ B
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    Omega = Omega.T @ Omega
    BetaOmegaBeta = beta @ Omega @ beta.T

    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    beta_penalty = - beta_tausq.T @ np.diag(BetaOmegaBeta)
    G_penalty = - G_eta * np.linalg.norm(G, ord=1)
    loss = log_likelihood + beta_penalty + G_penalty

    # Manual gradients
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T

    dLogL_dd = np.sum(y_minus_lambdadt, axis=1)
    d_plus = d + (1/smooth) * dLogL_dd
    dlogL_dG = y_minus_lambdadt_times_B @ beta.T
    G_minus = G + (1/G_smooth) * dlogL_dG
    G_plus = np.maximum(np.abs(G_minus) - G_eta/G_smooth, 0) * np.sign(G_minus)
    dlogL_dbeta = (G.T @ y_minus_lambdadt_times_B -
                   2 * beta_tausq[:, np.newaxis] * beta @ Omega)
    beta_minus = beta + (1/smooth) * dlogL_dbeta
    beta_plus = np.maximum(beta_minus, 0)

    result = {
        "dLogL_dd": dLogL_dd,
        "d_plus": d_plus,
        "dlogL_dG": dlogL_dG,
        "G_plus": G_plus,
        "dlogL_dbeta": dlogL_dbeta,
        "beta_plus": beta_plus,
        "dlogL_beta_tausq": 0,
        "loss": loss,
        "log_likelihood": log_likelihood,
        "beta_penalty": beta_penalty,
        "G_penalty": G_penalty
    }

    return result


