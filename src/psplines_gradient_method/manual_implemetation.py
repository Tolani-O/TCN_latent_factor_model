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


def log_obj(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt):
    J = np.ones_like(Y)
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G @ beta @ B
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    Omega = Omega.T @ Omega
    BetaOmegaBeta = beta @ Omega @ beta.T

    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    beta_penalty = - tau_beta * np.sum(np.diag(BetaOmegaBeta))
    G_penalty = - tau_G * np.linalg.norm(G, ord=1)
    d_penalty = - tau_d * d.T @ d
    loss = log_likelihood + beta_penalty + G_penalty + d_penalty

    # Manual gradients
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T

    dLogL_dd = np.sum(y_minus_lambdadt, axis=1) - 2 * tau_d * d
    d_plus = d + (1/smooth_d) * dLogL_dd
    dlogL_dG = y_minus_lambdadt_times_B @ beta.T
    G_minus = G + (1/smooth_G) * dlogL_dG
    G_plus = np.maximum(np.abs(G_minus) - tau_G/smooth_G, 0) * np.sign(G_minus)
    dlogL_dbeta = G.T @ y_minus_lambdadt_times_B - 2 * tau_beta * beta @ Omega
    beta_minus = beta + smooth_beta * dlogL_dbeta
    beta_plus = np.maximum(beta_minus, 0)

    result = {
        "dLogL_dd": dLogL_dd,
        "d_plus": d_plus,
        "dlogL_dG": dlogL_dG,
        "G_plus": G_plus,
        "dlogL_dbeta": dlogL_dbeta,
        "beta_plus": beta_plus,
        "loss": loss,
        "log_likelihood": log_likelihood,
        "d_penalty": d_penalty,
        "beta_penalty": beta_penalty,
        "G_penalty": G_penalty
    }

    return result


def backtracking_line_search(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, smooth_G, smooth_d, dt,
                             alpha=0.3, beta_factor=0.8):
    # Current function value and gradient
    t = 1
    current_values = log_obj(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, t, smooth_G, smooth_d, dt)
    f_curr = current_values["loss"]
    grad_curr = current_values["dlogL_dbeta"]
    beta_next = current_values["beta_plus"]
    gen_grad_curr = (beta - beta_next) / t
    next_values = log_obj(Y, B, d, G, beta_next, Omega, tau_beta, tau_G, tau_d, t, smooth_G, smooth_d, dt)
    f_next = next_values["loss"]
    # grad_next = next_values["dlogL_dbeta"]
    # beta_next_next = next_values["beta_plus"]

    # initialize step size
    while True:
        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if f_next >= f_curr + alpha * t * np.sum(grad_curr * gen_grad_curr) + alpha * t * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro')**2:
            break
        t *= beta_factor

        next_values = log_obj(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, t, smooth_G, smooth_d, dt)
        beta_next = next_values["beta_plus"]
        grad_curr = current_values["dlogL_dbeta"]
        gen_grad_curr = (beta - beta_next) / t
        next_values = log_obj(Y, B, d, G, beta_next, Omega, tau_beta, tau_G, tau_d, t, smooth_G, smooth_d, dt)
        f_next = next_values["loss"]



    current_values["beta_plus"] = beta_next
    current_values["loss"] = f_next

    return current_values
