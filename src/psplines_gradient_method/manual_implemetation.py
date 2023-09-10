import numpy as np
from numpy.linalg import det
from sklearn.isotonic import IsotonicRegression

from src.psplines_gradient_method.general_functions import create_precision_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bspline_matrix


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


def log_obj_with_backtracking_line_search_and_time_warping(
        Y, J, B_func_n, B_func_nminus1, knots_1, knots_2, V,
        d, G_star, mask_G, beta, mask_beta, I_beta_P, I_beta_L, psi, mask_psi, J_psi, Omega_beta, Omega_psi,
        tau_psi, tau_beta, tau_G, dt=1, alpha=0.3, max_iters=4,
        psi_factor=1e-2, beta_factor=1e-2, G_factor=1e-2, d_factor=1e-2):

    smooth_psi, smooth_beta, smooth_G, smooth_d = 1, 1, 1, 1
    K = Y.shape[0]
    time_matrix = psi @ V
    B_psi = generate_bspline_matrix(B_func_n, time_matrix)
    V_star = np.repeat(V, K, axis=0)
    beta_star = np.kron(np.eye(K), beta)
    diagdJ = d[:, np.newaxis] * J
    GStar_BetaStar = G_star @ beta_star
    Omega_beta = Omega_beta.T @ Omega_beta
    Omega_psi = Omega_psi.T @ Omega_psi
    diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    psi_penalty = - tau_psi * np.sum(np.diag(psi @ Omega_psi @ psi.T))
    beta_penalty = - tau_beta * np.sum(np.diag(beta @ Omega_beta @ beta.T))
    G_penalty = - tau_G * np.linalg.norm(G_star, ord=1)
    loss = log_likelihood + psi_penalty + beta_penalty + G_penalty

    # smooth_psi
    ct = 0
    B_psi_nminus1_1 = generate_bspline_matrix(B_func_nminus1, time_matrix)
    B_psi_nminus1_2 = np.zeros_like(B_psi_nminus1_1)
    P = len(B_func_nminus1)
    for i in range(K):
        # Copy rows from A to B with a shift
        B_psi_nminus1_2[i * P:((i + 1) * P) - 1, :] = B_psi_nminus1_1[(i * P) + 1:((i + 1) * P), :]
    y_minus_lambdadt_star = np.vstack([Y - lambda_del_t] * V.shape[0])
    dlogL_dpsi = (((GStar_BetaStar @
                   ((np.vstack([knots_1] * K) * B_psi_nminus1_1) - (np.vstack([knots_2] * K) * B_psi_nminus1_2)) @
                   (V_star * y_minus_lambdadt_star).T) * mask_psi) @
                  J_psi) - 2 * tau_psi * psi @ Omega_psi
    iso_reg = IsotonicRegression()
    x = np.arange(psi.shape[1])
    while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
        psi_minus = psi + smooth_psi * dlogL_dpsi
        psi_plus = np.zeros_like(psi_minus)
        for i in range(psi_minus.shape[0]):
            psi_plus[i, :] = iso_reg.fit(x, psi_minus[i, :]).predict(x)
        time_matrix = psi_plus @ V
        B_psi_plus = generate_bspline_matrix(B_func_n, time_matrix)

        gen_grad_curr = (psi_plus - psi) / smooth_psi
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi_plus
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

        log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(psi_plus @ Omega_psi @ psi_plus.T))
        loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if (loss_next >= loss + alpha * smooth_psi * np.sum(dlogL_dpsi * gen_grad_curr) +
                alpha * smooth_psi * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
            break
        smooth_psi *= psi_factor
        ct += 1

    loss_psi = loss_next
    # log_likelihood_psi = log_likelihood
    ct_psi = ct
    if ct == max_iters:
        loss_psi = loss
        psi_plus = psi
        smooth_psi = 0
        ct_psi = np.inf

    # smooth_beta
    ct = 0
    time_matrix = psi_plus @ V
    B_psi_plus = generate_bspline_matrix(B_func_n, time_matrix)
    diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi_plus
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    dlogL_dbeta = (G_star @ I_beta_L).T @ (((Y - lambda_del_t) @ B_psi_plus.T) * mask_beta) @ I_beta_P - 2 * tau_beta * beta @ Omega_beta
    while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
        beta_minus = beta + smooth_beta * dlogL_dbeta
        beta_plus = np.maximum(beta_minus, 0)

        gen_grad_curr = (beta_plus - beta) / smooth_beta
        diagdJ_plus_GBetaB = diagdJ + G_star @ np.kron(np.eye(Y.shape[0]), beta_plus) @ B_psi_plus
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

        log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
        beta_penalty = - tau_beta * np.sum(np.diag(beta_plus @ Omega_beta @ beta_plus.T))
        loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if (loss_next >= loss_psi + alpha * smooth_beta * np.sum(dlogL_dbeta * gen_grad_curr) +
                alpha * smooth_beta * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
            break
        smooth_beta *= beta_factor
        ct += 1

    loss_beta = loss_next
    #log_likelihood_beta = log_likelihood
    ct_beta = ct
    if ct == max_iters:
        loss_beta = loss_psi
        beta_plus = beta
        smooth_beta = 0
        ct_beta = np.inf

    # smooth_G
    ct = 0
    betaStarPlus_BPsiPlus = np.kron(np.eye(Y.shape[0]), beta_plus) @ B_psi_plus
    diagdJ_plus_GBetaB = diagdJ + G_star @ betaStarPlus_BPsiPlus
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    dlogL_dG_star = ((Y - lambda_del_t) @ betaStarPlus_BPsiPlus.T) * mask_G
    while ct < max_iters:
        G_star_minus = G_star + smooth_G * dlogL_dG_star
        G_star_plus = np.maximum(np.abs(G_star_minus) - tau_G * smooth_G, 0) * np.sign(G_star_minus)

        gen_grad_curr = (G_star_plus - G_star) / smooth_G
        diagdJ_plus_GBetaB = diagdJ + G_star_plus @ betaStarPlus_BPsiPlus
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

        log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
        G_penalty = - tau_G * np.linalg.norm(G_star_plus, ord=1)
        loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if (loss_next >= loss_beta + alpha * smooth_G * np.sum(dlogL_dG_star * gen_grad_curr) +
                alpha * smooth_G * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
            break
        smooth_G *= G_factor
        ct += 1

    loss_G = loss_next
    # log_likelihood_G = log_likelihood
    ct_G = ct
    if ct == max_iters:
        loss_G = loss_beta
        G_star_plus = G_star
        smooth_G = 0
        ct_G = np.inf

    # smooth_d
    ct = 0
    GStarPlus_betaStarPlus_BPsiPlus = G_star_plus @ betaStarPlus_BPsiPlus
    diagdJ_plus_GBetaB = diagdJ + GStarPlus_betaStarPlus_BPsiPlus
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    dLogL_dd = np.sum(Y - lambda_del_t, axis=1)
    while ct < max_iters:
        d_plus = d + smooth_d * dLogL_dd

        diagdJ_plus_GBetaB = d_plus[:, np.newaxis] * J + GStarPlus_betaStarPlus_BPsiPlus
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt

        log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
        loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

        # Armijo condition, using Frobenius norm for matrices, but for maximization
        if loss_next >= loss_G + alpha * smooth_d * np.sum(dLogL_dd * dLogL_dd):
            break
        smooth_d *= d_factor
        ct += 1

    loss_d = loss_next
    # log_likelihood_d = log_likelihood
    ct_d = ct
    if ct == max_iters:
        loss_d = loss_G
        d_plus = d
        smooth_d = 0
        ct_d = np.inf

    result = {
        "dlogL_dpsi": dlogL_dpsi,
        "psi_plus": psi_plus,
        "psi_loss_increase": loss_psi - loss,
        "smooth_psi": smooth_psi,
        "iters_psi": ct_psi,
        "dLogL_dd": dLogL_dd,
        "d_plus": d_plus,
        "d_loss_increase": loss_d - loss_G,
        "smooth_d": smooth_d,
        "iters_d": ct_d,
        "dlogL_dG": dlogL_dG_star,
        "G_star_plus": G_star_plus,
        "G_loss_increase": loss_G - loss_beta,
        "smooth_G": smooth_G,
        "iters_G": ct_G,
        "dlogL_dbeta": dlogL_dbeta,
        "beta_plus": beta_plus,
        "beta_loss_increase": loss_beta - loss,
        "smooth_beta": smooth_beta,
        "iters_beta": ct_beta,
        "loss": loss,
        "log_likelihood": log_likelihood,
        "beta_penalty": beta_penalty,
        "G_penalty": G_penalty,
        "psi_penalty": psi_penalty
    }

    return result
