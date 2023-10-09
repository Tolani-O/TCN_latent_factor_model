import numpy as np
import multiprocessing as mp
from src.psplines_gradient_method.general_functions import create_first_diff_matrix, create_masking_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bspline_functions, generate_bspline_matrix, \
    bspline_deriv_multipliers


class SpikeTrainModel:
    def __init__(self, Y, time):
        # variables
        self.chi = None
        self.gamma = None
        self.d = None
        self.alpha = None

        # parameters
        self.Y = Y
        self.time = time
        self.mask_G = None
        self.J = None
        self.mask_beta = None
        self.I_beta_P = None
        self.I_beta_L = None
        self.alpha_prime_multiply = None
        self.alpha_prime_add = None
        self.mask_psi = None
        self.J_psi = None
        self.U_psi = None
        self.V = None
        self.B_func_n = None
        self.Omega_beta = None
        self.Omega_psi = None
        self.degree = None

    def initialize_for_time_warping(self, L, degree):

        # parameters
        # time warping and latent factor b-spline functions. May need to separate these out later
        self.degree = degree
        self.B_func_n = generate_bspline_functions(self.time, self.degree)
        self.B_func_nminus1 = generate_bspline_functions(self.time, self.degree, True)  # for psi derivatives
        self.knots_1, self.knots_2 = bspline_deriv_multipliers(self.time, self.degree)  # for psi derivatives

        time_matrix = self.time[np.newaxis, :]
        # time warping b-spline matrix. Coefficients would be from psi
        self.V = generate_bspline_matrix(self.B_func_n, time_matrix)
        K = self.Y.shape[0]
        P = len(self.B_func_n)
        Q = self.V.shape[0]  # will be equal to P now
        self.mask_G = create_masking_matrix(K, L)
        self.mask_beta = create_masking_matrix(K, P)
        self.I_beta_P = np.vstack([np.eye(P)] * K)
        self.I_beta_L = np.vstack([np.eye(L)] * K)
        self.J = np.ones_like(self.Y)
        self.alpha_prime_multiply = np.eye(Q)
        self.alpha_prime_multiply[0, 0] = 0
        self.alpha_prime_multiply[1, 1] = 0
        self.alpha_prime_add = np.zeros((K, Q))
        self.alpha_prime_add[:, 1] = 1
        self.mask_psi = np.hstack([np.eye(K)] * Q)
        self.J_psi = create_masking_matrix(Q, K).T
        self.U_psi = np.triu(np.ones((Q, Q)))
        self.Omega_beta = create_first_diff_matrix(P)
        self.Omega_beta = self.Omega_beta.T @ self.Omega_beta
        self.Omega_psi = create_first_diff_matrix(Q)
        self.Omega_psi = self.Omega_psi.T @ self.Omega_psi

        # variables
        np.random.seed(0)
        self.chi = np.random.rand(K, L)
        np.random.seed(0)
        self.gamma = np.random.rand(L, P)
        np.random.seed(0)
        self.d = np.random.rand(K)
        np.random.seed(0)
        self.alpha = np.random.rand(K, Q)

        return self

    def log_obj_with_backtracking_line_search_and_time_warping(self, tau_psi, tau_beta,
                                                               alpha_factor=1e-2, gamma_factor=1e-2,
                                                               G_factor=1e-2, d_factor=1e-2,
                                                               alpha=0.3, max_iters=4):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)
        K = self.Y.shape[0]
        P = len(self.B_func_n)
        Q = self.V.shape[0]
        L = self.gamma.shape[0]

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add
        psi = exp_alpha_c @ self.U_psi  # variable
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # variable, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # variable
        diagdJ = self.d[:, np.newaxis] * self.J  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1/np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        G_star = (G @ self.I_beta_L.T) * self.mask_G  # variable
        GStar_BetaStar = G_star @ np.kron(np.eye(K), beta)  # variable
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))
        loss = log_likelihood + psi_penalty + beta_penalty
        loss_0 = loss

        # smooth_gamma
        # print('Optimizing gamma')
        ct = 0
        learning_rate = 1
        dlogL_dgamma = beta * (G.T @
                               (((self.Y - lambda_del_t) @ B_psi.T) * self.mask_beta) @
                               self.I_beta_P - 2 * tau_beta * beta @ self.Omega_beta)
        while ct < max_iters:
            gamma_plus = self.gamma + learning_rate * dlogL_dgamma

            # set up variables to compute loss
            beta = np.exp(gamma_plus)
            GStar_BetaStar = G_star @ np.kron(np.eye(K), beta)
            diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))
            loss_next = log_likelihood + psi_penalty + beta_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dgamma, ord='fro') ** 2:
                break
            learning_rate *= gamma_factor
            ct += 1

        if ct < max_iters:
            ct_gamma = ct
            smooth_gamma = learning_rate
            loss = loss_next
            self.gamma = gamma_plus
        else:
            ct_gamma = np.inf
            smooth_gamma = 0
        loss_gamma = loss

        # set up variables to compute loss in next round
        # exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add  # didnt change
        # psi = exp_alpha_c @ self.U_psi  # didnt change
        # psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # didnt change, called \psi' in the document
        # time_matrix = max(self.time) * (psi_norm @ self.V)  # didnt change
        # B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # didnt change
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        beta = np.exp(self.gamma)  # now fixed
        # exp_chi = np.exp(self.chi)  # didnt change
        # G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # didnt change
        # G_star = (G @ self.I_beta_L.T) * self.mask_G  # didnt change
        GStar_BetaStar = G_star @ np.kron(np.eye(K), beta)  # variable
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi  # now fixed
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))

        # smooth_alpha
        # print('Optimizing alpha')
        ct = 0
        learning_rate = 1
        B_psi_nminus1_1 = generate_bspline_matrix(self.B_func_nminus1, time_matrix)  # variable
        B_psi_nminus1_2 = np.zeros_like(B_psi_nminus1_1)  # variable
        for i in range(K):
            # Copy rows from A to B with a shift
            B_psi_nminus1_2[i * P:(((i + 1) * P) - 1), :] = B_psi_nminus1_1[((i * P) + 1):((i + 1) * P), :]
        y_minus_lambdadt_star = np.vstack([self.Y - lambda_del_t] * Q)
        U_psi_star = np.repeat(self.U_psi, K, axis=0)
        psi_norm_star = np.vstack([psi_norm] * Q)
        psi_norm_1_exp_alpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c
        U_star_minus_psi_star = U_psi_star - psi_norm_star
        # alpha gradient
        dlogL_dalpha = (self.degree * max(self.time) * (((GStar_BetaStar @
                    ((np.vstack([self.knots_1] * K) * B_psi_nminus1_1) - (np.vstack([self.knots_2] * K) * B_psi_nminus1_2)) @
                    (psi_norm_1_exp_alpha.T.flatten()[:, np.newaxis] * (U_star_minus_psi_star @ self.V) * y_minus_lambdadt_star).T) * self.mask_psi) @
                                    self.J_psi) - 2 * tau_psi * psi_norm_1_exp_alpha *
                                    np.reshape(np.diag(psi_norm_star @ self.Omega_psi @ U_star_minus_psi_star.T), (K, Q), order='F'))
        while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
            alpha_plus = self.alpha + learning_rate * dlogL_dalpha

            # set up variables to compute loss
            exp_alpha_c = (np.exp(alpha_plus) @ self.alpha_prime_multiply) + self.alpha_prime_add
            psi = exp_alpha_c @ self.U_psi  # variable
            psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # variable, called \psi' in the document
            time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
            B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)
            diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))
            loss_next = log_likelihood + psi_penalty + beta_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dalpha, ord='fro')**2:
                break
            learning_rate *= alpha_factor
            ct += 1

        if ct < max_iters:
            ct_alpha = ct
            smooth_alpha = learning_rate
            loss = loss_next
            self.alpha = alpha_plus
        else:
            ct_alpha = np.inf
            smooth_alpha = 0
        loss_alpha = loss

        # set up variables to compute loss in next round
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add  # now fixed
        psi = exp_alpha_c @ self.U_psi  # now fixed
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # now fixed
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        # beta = np.exp(self.gamma)  # now fixed
        # exp_chi = np.exp(self.chi)  # didnt change
        # G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # didnt change
        # G_star = (G @ self.I_beta_L.T) * self.mask_G  # didnt change
        # GStar_BetaStar = G_star @ np.kron(np.eye(K), beta)  # didnt change
        betaStar_BPsi = np.kron(np.eye(K), beta) @ B_psi  # now fixed
        diagdJ_plus_GBetaB = diagdJ + G_star @ betaStar_BPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))

        # smooth_chi
        ct = 0
        learning_rate = 1
        GStar_BetaStar_star = np.vstack([GStar_BetaStar] * L)
        beta_star = beta @ self.I_beta_P.T
        col_slices = [slice(i * P, (i + 1) * P) for i in range(K)]
        betaStar_star = np.zeros_like(GStar_BetaStar_star)
        for i, col_slice in enumerate(col_slices):
            betaStar_star[i::K, col_slice] = beta_star[:, col_slice]
        GStarBetaStar_minus_betaStar = GStar_BetaStar_star - betaStar_star
        dlogL_dchi = (((self.Y - lambda_del_t) @ B_psi.T @ (G.flatten()[:, np.newaxis] * GStarBetaStar_minus_betaStar).T) * self.mask_G) @ self.I_beta_L
        while ct < max_iters:
            chi_plus = self.chi + learning_rate * dlogL_dchi

            # set up variables to compute loss
            exp_chi = np.exp(chi_plus)  # variable
            G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
            G_star = (G @ self.I_beta_L.T) * self.mask_G  # variable
            diagdJ_plus_GBetaB = diagdJ + G_star @ betaStar_BPsi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            loss_next = log_likelihood + psi_penalty + beta_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if (loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dchi, ord='fro') ** 2):
                break
            learning_rate *= G_factor
            ct += 1

        if ct < max_iters:
            ct_chi = ct
            smooth_chi = learning_rate
            loss = loss_next
            self.chi = chi_plus
        else:
            ct_chi = np.inf
            smooth_chi = 0
        loss_chi = loss

        # set up variables to compute loss in next round
        # exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add  # now fixed
        # psi = exp_alpha_c @ self.U_psi  # now fixed
        # psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        # time_matrix = max(self.time) * (psi_norm @ self.V)  # now fixed
        # B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        # beta = np.exp(self.gamma)  # now fixed
        exp_chi = np.exp(self.chi)  # now fixed
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # now fixed
        G_star = (G @ self.I_beta_L.T) * self.mask_G  # now fixed
        GStar_BetaStar_BPsi = G_star @ betaStar_BPsi  # now fixed
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar_BPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable

        # smooth_d
        ct = 0
        learning_rate = 1
        dlogL_dd = np.sum(self.Y - lambda_del_t, axis=1)
        while ct < max_iters:
            d_plus = self.d + learning_rate * dlogL_dd

            # set up variables to compute loss
            diagdJ_plus_GBetaB = d_plus[:, np.newaxis] * self.J + GStar_BetaStar_BPsi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            loss_next = log_likelihood + psi_penalty + beta_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dd * dlogL_dd):
                break
            learning_rate *= d_factor
            ct += 1

        if ct < max_iters:
            ct_d = ct
            smooth_d = learning_rate
            loss = loss_next
            self.d = d_plus
        else:
            ct_d = np.inf
            smooth_d = 0
        loss_d = loss

        result = {
            "dlogL_dgamma": dlogL_dgamma,
            "gamma_loss_increase": loss_gamma - loss_0,
            "smooth_gamma": smooth_gamma,
            "iters_gamma": ct_gamma,
            "dlogL_dalpha": dlogL_dalpha,
            "alpha_loss_increase": loss_alpha - loss_gamma,
            "smooth_alpha": smooth_alpha,
            "iters_alpha": ct_alpha,
            "dlogL_dchi": dlogL_dchi,
            "chi_loss_increase": loss_chi - loss_alpha,
            "smooth_chi": smooth_chi,
            "iters_chi": ct_chi,
            "dlogL_dd": dlogL_dd,
            "d_loss_increase": loss_d - loss_chi,
            "smooth_d": smooth_d,
            "iters_d": ct_d,
            "loss": loss,
            "beta_penalty": beta_penalty,
            "psi_penalty": psi_penalty
        }

        return result

    def compute_loss_time_warping(self, tau_psi, tau_beta):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)
        K = self.Y.shape[0]
        Q = self.V.shape[0]

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add
        psi = exp_alpha_c @ self.U_psi  # variable
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # variable, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
        # time_matrix = np.repeat(self.time[np.newaxis, :], K, axis=0); tau_psi = 0
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        G_star = (G @ self.I_beta_L.T) * self.mask_G  # variable
        diagdJ_plus_GBetaB = self.d[:, np.newaxis] * self.J + G_star @ np.kron(np.eye(K), beta) @ B_psi
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))
        loss = log_likelihood + psi_penalty + beta_penalty

        result = {
            "loss": loss,
            "log_likelihood": log_likelihood,
            "psi_penalty": psi_penalty,
            "beta_penalty": beta_penalty,
        }
        return result

    def compute_analytical_grad_time_warping(self, tau_psi, tau_beta):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)
        K = self.Y.shape[0]
        P = len(self.B_func_n)
        Q = self.V.shape[0]
        L = self.gamma.shape[0]

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add  # now fixed
        psi = exp_alpha_c @ self.U_psi  # now fixed
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # now fixed
        # time_matrix = np.repeat(self.time[np.newaxis, :], K, axis=0); tau_psi = 0
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        G_star = (G @ self.I_beta_L.T) * self.mask_G  # variable
        GStar_BetaStar = G_star @ np.kron(np.eye(K), beta)  # variable
        diagdJ_plus_GBetaB = self.d[:, np.newaxis] * self.J + GStar_BetaStar @ B_psi
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
        # compute gradients
        B_psi_nminus1_1 = generate_bspline_matrix(self.B_func_nminus1, time_matrix)  # variable
        B_psi_nminus1_2 = np.zeros_like(B_psi_nminus1_1)  # variable
        for i in range(K):
            # Copy rows from A to B with a shift
            B_psi_nminus1_2[i * P:(((i + 1) * P) - 1), :] = B_psi_nminus1_1[((i * P) + 1):((i + 1) * P), :]
        y_minus_lambdadt_star = np.vstack([self.Y - lambda_del_t] * Q)
        U_psi_star = np.repeat(self.U_psi, K, axis=0)
        psi_norm_star = np.vstack([psi_norm] * Q)
        psi_norm_1_exp_alpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c
        U_star_minus_psi_star = U_psi_star - psi_norm_star
        GStar_BetaStar_star = np.vstack([GStar_BetaStar] * L)
        betaStar_star = np.repeat(np.kron(np.eye(K), beta), K, axis=0)
        GStarBetaStar_minus_betaStar = GStar_BetaStar_star - betaStar_star
        # gradients
        dlogL_dalpha = (self.degree * max(self.time) * (((GStar_BetaStar @
                                ((np.vstack([self.knots_1] * K) * B_psi_nminus1_1) - (np.vstack([self.knots_2] * K) * B_psi_nminus1_2)) @
                                (psi_norm_1_exp_alpha.T.flatten()[:, np.newaxis] * (U_star_minus_psi_star @ self.V) * y_minus_lambdadt_star).T) * self.mask_psi) @
                                       self.J_psi) - 2 * tau_psi * psi_norm_1_exp_alpha *
                        np.reshape(np.diag(psi_norm_star @ self.Omega_psi @ U_star_minus_psi_star.T), (K, Q), order='F'))

        dlogL_dgamma = beta * ((self.G_star @ self.I_beta_L).T @
                               (((self.Y - lambda_del_t) @ B_psi.T) * self.mask_beta) @
                               self.I_beta_P - 2 * tau_beta * beta @ self.Omega_beta)

        dlogL_dchi = (((self.Y - lambda_del_t) @ B_psi.T @ (G.flatten()[:, np.newaxis] * GStarBetaStar_minus_betaStar).T) * self.mask_G) @ self.I_beta_L

        dlogL_dd = np.sum(self.Y - lambda_del_t, axis=1)

        return dlogL_dalpha, dlogL_dgamma, dlogL_dchi, dlogL_dd

    def compute_grad_chunk(self, name, variable, i, eps, tau_psi, tau_beta, loss):

        J = variable.shape[1]
        grad_chunk = np.zeros(J)

        print(f'Starting {name} gradient chunk at row: {i}')

        for j in range(J):
            orig = variable[i, j]
            variable[i, j] = orig + eps
            if name == 'd':
                self.d[j] = orig + eps
            loss_result = self.compute_loss_time_warping(tau_psi, tau_beta)
            loss_eps = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
            grad_chunk[j] = (loss_eps - loss) / eps
            variable[i, j] = orig
            if name == 'd':
                self.d[j] = orig

        print(f'Completed {name} gradient chunk at row: {i}')

        return grad_chunk

    def compute_numerical_grad_time_warping_parallel(self, tau_psi, tau_beta, tau_G):

        eps = 1e-4
        # define parameters
        K = self.Y.shape[0]
        L = self.gamma.shape[0]

        loss_result = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
        loss = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
        pool = mp.Pool()

        # alpha gradient
        alpha_async_results = [pool.apply_async(self.compute_grad_chunk, args=('alpha', self.alpha, k, eps, tau_psi, tau_beta, tau_G, loss)) for k in range(K)]

        # gamma gradient
        gamma_async_results = [pool.apply_async(self.compute_grad_chunk, args=('gamma', self.gamma, l, eps, tau_psi, tau_beta, tau_G, loss)) for l in range(L)]

        # chi gradient
        chi_async_results = [pool.apply_async(self.compute_grad_chunk, args=('chi', self.chi, k, eps, tau_psi, tau_beta, tau_G, loss)) for k in range(K)]

        # d gradient
        d_grad = pool.apply_async(self.compute_grad_chunk, args=('d', self.d[np.newaxis, :], 0, eps, tau_psi, tau_beta, tau_G, loss))

        pool.close()
        pool.join()

        alpha_grad = np.vstack([r.get() for r in alpha_async_results])
        gamma_grad = np.vstack([r.get() for r in gamma_async_results])
        chi_grad = np.vstack([r.get() for r in chi_async_results])
        d_grad = d_grad.get()

        return alpha_grad, gamma_grad, chi_grad, d_grad
