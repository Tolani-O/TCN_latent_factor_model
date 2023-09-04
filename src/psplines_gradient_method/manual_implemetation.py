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

    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    beta_penalty = - tau_beta * np.sum(np.diag(beta @ Omega @ beta.T))
    G_penalty = - tau_G * np.linalg.norm(G, ord=1)
    d_penalty = - tau_d * d.T @ d
    loss = log_likelihood + beta_penalty + G_penalty + d_penalty

    # Manual gradients
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T

    dLogL_dd = np.sum(y_minus_lambdadt, axis=1) - 2 * tau_d * d
    d_plus = d + (1 / smooth_d) * dLogL_dd
    dlogL_dG = y_minus_lambdadt_times_B @ beta.T
    G_minus = G + (1 / smooth_G) * dlogL_dG
    G_plus = np.maximum(np.abs(G_minus) - tau_G / smooth_G, 0) * np.sign(G_minus)
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


def log_obj_with_backtracking_line_search(Y, B, d, G, beta, Omega, tau_beta, tau_G, dt=1,
                                          alpha=0.3, beta_factor=1e-2, G_factor=1e-2, d_factor=1e-2, max_iters=4):
    smooth_beta, smooth_G, smooth_d = 1, 1, 1
    J = np.ones_like(Y)
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G @ beta @ B
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    Omega = Omega.T @ Omega

    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    beta_penalty = - tau_beta * np.sum(np.diag(beta @ Omega @ beta.T))
    G_penalty = - tau_G * np.linalg.norm(G, ord=1)
    loss = log_likelihood + beta_penalty + G_penalty

    # Manual gradients
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T

    dLogL_dd = np.sum(y_minus_lambdadt, axis=1)
    dlogL_dbeta = G.T @ y_minus_lambdadt_times_B - 2 * tau_beta * beta @ Omega
    dlogL_dG = y_minus_lambdadt_times_B @ beta.T

    # smooth_beta
    ct = 0
    while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
        beta_minus = beta + smooth_beta * dlogL_dbeta
        beta_plus = np.maximum(beta_minus, 0)

        gen_grad_curr = (beta_plus - beta) / smooth_beta
        diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G @ beta_plus @ B
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

        log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
        beta_penalty = - tau_beta * np.sum(np.diag(beta_plus @ Omega @ beta_plus.T))
        loss_next = log_likelihood + beta_penalty + G_penalty

        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if (loss_next >= loss + alpha * smooth_beta * np.sum(dlogL_dbeta * gen_grad_curr) +
                alpha * smooth_beta * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
            break
        smooth_beta *= beta_factor
        ct += 1

    loss_beta = loss_next
    log_likelihood_beta = log_likelihood
    ct_beta = ct
    if ct == max_iters:
        beta_plus = beta
        smooth_beta = 0
        ct_beta = np.inf

    # smooth_G
    ct = 0
    while ct < max_iters:
        G_minus = G + smooth_G * dlogL_dG
        G_plus = np.maximum(np.abs(G_minus) - tau_G * smooth_G, 0) * np.sign(G_minus)

        gen_grad_curr = (G_plus - G) / smooth_G
        diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G_plus @ beta_plus @ B
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

        log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
        G_penalty = - tau_G * np.linalg.norm(G_plus, ord=1)
        loss_next = log_likelihood + beta_penalty + G_penalty

        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if (loss_next >= loss_beta + alpha * smooth_G * np.sum(dlogL_dG * gen_grad_curr) +
                alpha * smooth_G * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
            break
        smooth_G *= G_factor
        ct += 1

    loss_G = loss_next
    log_likelihood_G = log_likelihood
    ct_G = ct
    if ct == max_iters:
        G_plus = G
        smooth_G = 0
        ct_G = np.inf

    # smooth_d
    ct = 0
    while ct < max_iters:
        d_plus = d + smooth_d * dLogL_dd

        diagdJ_plus_GBetaB = d_plus[:, np.newaxis] * J + G_plus @ beta_plus @ B
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

        log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
        loss_next = log_likelihood + beta_penalty + G_penalty

        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if loss_next >= loss_G + alpha * smooth_d * np.sum(dlogL_dG * dlogL_dG):
            break
        smooth_d *= d_factor
        ct += 1

    loss_d = loss_next
    log_likelihood_d = log_likelihood
    ct_d = ct
    if ct == max_iters:
        d_plus = d
        smooth_d = 0
        ct_d = np.inf

    result = {
        "dLogL_dd": dLogL_dd,
        "d_plus": d_plus,
        "d_loss_increase": loss_d - loss_G,
        "d_likelihood_next": log_likelihood_d,
        "smooth_d": smooth_d,
        "iters_d": ct_d,
        "dlogL_dG": dlogL_dG,
        "G_plus": G_plus,
        "G_loss_increase": loss_G - loss_beta,
        "G_likelihood_next": log_likelihood_G,
        "smooth_G": smooth_G,
        "iters_G": ct_G,
        "dlogL_dbeta": dlogL_dbeta,
        "beta_plus": beta_plus,
        "beta_loss_increase": loss_beta - loss,
        "beta_likelihood_next": log_likelihood_beta,
        "smooth_beta": smooth_beta,
        "iters_beta": ct_beta,
        "loss": loss,
        "log_likelihood": log_likelihood,
        "beta_penalty": beta_penalty,
        "G_penalty": G_penalty
    }

    return result
