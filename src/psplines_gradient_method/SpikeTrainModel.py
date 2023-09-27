import numpy as np
import multiprocessing as mp
from src.psplines_gradient_method.general_functions import create_first_diff_matrix, create_masking_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bspline_functions, generate_bspline_matrix, \
    bspline_deriv_multipliers


class SpikeTrainModel:
    def __init__(self, Y, time):
        # variables
        self.G_star = None
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
        self.G_star = np.random.rand(K, K * L) * self.mask_G
        np.random.seed(0)
        self.gamma = np.random.rand(L, P)
        np.random.seed(0)
        self.d = np.random.rand(K)
        np.random.seed(0)
        self.alpha = np.random.rand(K, Q)

        return self

    def log_obj_with_backtracking_line_search_and_time_warping(self, tau_psi, tau_beta, tau_G,
                                                               alpha_factor=1e-2, gamma_factor=1e-2,
                                                               G_factor=1e-2, d_factor=1e-2,
                                                               alpha=0.3, max_iters=4):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)
        K = self.Y.shape[0]
        P = len(self.B_func_n)
        Q = self.V.shape[0]

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add
        psi = exp_alpha_c @ self.U_psi  # variable
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # variable, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # variable
        diagdJ = self.d[:, np.newaxis] * self.J  # variable
        beta = np.exp(self.gamma)  # variable
        GStar_BetaStar = self.G_star @ np.kron(np.eye(K), beta)  # variable
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))
        G_penalty = - tau_G * np.linalg.norm(self.G_star, ord=1)
        loss = log_likelihood + psi_penalty + beta_penalty + G_penalty
        loss_0 = loss

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
        # psi gradient
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
            loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

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
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # now fixed
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        # beta = np.exp(self.gamma)  # didnt change
        # GStar_BetaStar = self.G_star @ np.kron(np.eye(K), beta)  # didnt change
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))

        # smooth_gamma
        # print('Optimizing gamma')
        ct = 0
        learning_rate = 1
        dlogL_dgamma = beta * ((self.G_star @ self.I_beta_L).T @
                       (((self.Y - lambda_del_t) @ B_psi.T) * self.mask_beta) @
                       self.I_beta_P - 2 * tau_beta * beta @ self.Omega_beta)
        while ct < max_iters:
            gamma_plus = self.gamma + learning_rate * dlogL_dgamma

            # set up variables to compute loss
            beta = np.exp(gamma_plus)
            GStar_BetaStar = self.G_star @ np.kron(np.eye(K), beta)
            diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))
            loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dgamma, ord='fro')**2:
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
        # exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add  # now fixed
        # psi = exp_alpha_c @ self.U_psi  # now fixed
        # psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        # time_matrix = max(self.time) * (psi_norm @ self.V)  # now fixed
        # B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        beta = np.exp(self.gamma)  # now fixed
        betaStar_BPsi = np.kron(np.eye(K), beta) @ B_psi  # now fixed
        diagdJ_plus_GBetaB = diagdJ + self.G_star @ betaStar_BPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))

        # smooth_G
        # print('Optimizing G_star')
        ct = 0
        learning_rate = 1
        dlogL_dG_star = ((self.Y - lambda_del_t) @ betaStar_BPsi.T) * self.mask_G
        while ct < max_iters:
            G_star_minus = self.G_star + learning_rate * dlogL_dG_star
            G_star_plus = np.maximum(np.abs(G_star_minus) - tau_G * learning_rate, 0) * np.sign(G_star_minus)
            gen_grad_curr = (G_star_plus - self.G_star) / learning_rate

            # set up variables to compute loss
            diagdJ_plus_GBetaB = diagdJ + G_star_plus @ betaStar_BPsi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            G_penalty = - tau_G * np.linalg.norm(G_star_plus, ord=1)
            loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if (loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dG_star * gen_grad_curr) +
                    alpha * learning_rate * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
                break
            learning_rate *= G_factor
            ct += 1

        if ct < max_iters:
            ct_G = ct
            smooth_G = learning_rate
            loss = loss_next
            self.G_star = G_star_plus
        else:
            ct_G = np.inf
            smooth_G = 0
        loss_G = loss

        # set up variables to compute loss in next round
        # exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add  # now fixed
        # psi = exp_alpha_c @ self.U_psi  # now fixed
        # psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        # time_matrix = max(self.time) * (psi_norm @ self.V)  # now fixed
        # B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        # beta = np.exp(self.gamma)  # now fixed
        # betaStar_BPsi = np.kron(np.eye(K), beta) @ B_psi  # now fixed
        GStar_BetaStar_BPsi = self.G_star @ betaStar_BPsi  # now fixed
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar_BPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        G_penalty = - tau_G * np.linalg.norm(self.G_star, ord=1)

        # smooth_d
        # print('Optimizing d')
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
            loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

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
            "dlogL_dalpha": dlogL_dalpha,
            "alpha_loss_increase": loss_alpha - loss_0,
            "smooth_alpha": smooth_alpha,
            "iters_alpha": ct_alpha,
            "dlogL_dgamma": dlogL_dgamma,
            "gamma_loss_increase": loss_gamma - loss_alpha,
            "smooth_gamma": smooth_gamma,
            "iters_gamma": ct_gamma,
            "dlogL_dG": dlogL_dG_star,
            "G_loss_increase": loss_G - loss_gamma,
            "smooth_G": smooth_G,
            "iters_G": ct_G,
            "dlogL_dd": dlogL_dd,
            "d_loss_increase": loss_d - loss_G,
            "smooth_d": smooth_d,
            "iters_d": ct_d,
            "loss": loss,
            "beta_penalty": beta_penalty,
            "G_penalty": G_penalty,
            "psi_penalty": psi_penalty
        }

        return result

    def compute_loss_time_warping(self, tau_psi, tau_beta, tau_G):
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
        diagdJ_plus_GBetaB = self.d[:, np.newaxis] * self.J + self.G_star @ np.kron(np.eye(K), beta) @ B_psi
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))
        G_penalty = - tau_G * np.linalg.norm(self.G_star, ord=1)
        loss = log_likelihood + psi_penalty + beta_penalty + G_penalty

        result = {
            "loss": loss,
            "log_likelihood": log_likelihood,
            "psi_penalty": psi_penalty,
            "beta_penalty": beta_penalty,
            "G_penalty": G_penalty
        }

        return result

    def compute_analytical_grad_time_warping(self, tau_psi, tau_beta):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)
        K = self.Y.shape[0]
        P = len(self.B_func_n)
        Q = self.V.shape[0]

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add  # now fixed
        psi = exp_alpha_c @ self.U_psi  # now fixed
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # now fixed
        # time_matrix = np.repeat(self.time[np.newaxis, :], K, axis=0); tau_psi = 0
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)
        beta = np.exp(self.gamma)  # variable
        GStar_BetaStar = self.G_star @ np.kron(np.eye(K), beta)  # variable
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
        # gradients
        dlogL_dalpha = (self.degree * max(self.time) * (((GStar_BetaStar @
                                ((np.vstack([self.knots_1] * K) * B_psi_nminus1_1) - (np.vstack([self.knots_2] * K) * B_psi_nminus1_2)) @
                                (psi_norm_1_exp_alpha.T.flatten()[:, np.newaxis] * (U_star_minus_psi_star @ self.V) * y_minus_lambdadt_star).T) * self.mask_psi) @
                                       self.J_psi) - 2 * tau_psi * psi_norm_1_exp_alpha *
                        np.reshape(np.diag(psi_norm_star @ self.Omega_psi @ U_star_minus_psi_star.T), (K, Q), order='F'))

        dlogL_dgamma = beta * ((self.G_star @ self.I_beta_L).T @
                               (((self.Y - lambda_del_t) @ B_psi.T) * self.mask_beta) @
                               self.I_beta_P - 2 * tau_beta * beta @ self.Omega_beta)

        dlogL_dG_star = ((self.Y - lambda_del_t) @ (np.kron(np.eye(K), beta) @ B_psi).T) * self.mask_G

        dlogL_dd = np.sum(self.Y - lambda_del_t, axis=1)

        return dlogL_dalpha, dlogL_dgamma, dlogL_dG_star, dlogL_dd

    def compute_grad_chunk(self, params):
        variable, i_start, i_end, eps, tau_psi, tau_beta, tau_G, loss = params
        J = variable.shape[1]
        J2 = 0
        grad_chunk = np.zeros((i_end - i_start, J))

        if variable.shape == self.alpha.shape:
            name = 'alpha'
        elif variable.shape == self.gamma.shape:
            name = 'gamma'
        elif variable.shape == self.G_star.shape:
            name = 'G_star'
            J = variable.shape[1] // variable.shape[0]
            J2 = J
        else:
            raise ValueError('Variable not recognized')

        print(f'Starting {name} gradient chunk. i_start: {i_start}, i_end: {i_end}')

        for i in range(i_start, i_end):
            for j in (np.arange(J) + (i * J2)):
                orig = variable[i, j]
                variable[i, j] = orig + eps
                loss_result = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
                loss_eps = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
                grad_chunk[i - i_start, j] = (loss_eps - loss) / eps
                variable[i, j] = orig

        print(f'Completed {name} gradient chunk. i_start: {i_start}, i_end: {i_end}')

        return grad_chunk

    def compute_numerical_grad_time_warping_parallel(self, tau_psi, tau_beta, tau_G):

        eps = 1e-3
        # define parameters
        K = self.Y.shape[0]
        L = self.gamma.shape[0]

        loss_result = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
        loss = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
        pool = mp.Pool()

        # alpha gradient
        chunk_size = K // mp.cpu_count()
        if chunk_size == 0:
            chunk_size = K
        params = []
        for k in range(0, K, chunk_size):
            k_start = k
            k_end = min(k + chunk_size, K)
            params.append((self.alpha, k_start, k_end, eps, tau_psi, tau_beta, tau_G, loss))

        results = pool.map(self.compute_grad_chunk, params)
        alpha_grad = np.concatenate([r for r in results])

        # gamma gradient
        chunk_size = L // mp.cpu_count()
        if chunk_size == 0:
            chunk_size = L
        params = []
        for l in range(0, L, chunk_size):
            l_start = l
            l_end = min(l + chunk_size, L)
            params.append((self.gamma, l_start, l_end, eps, tau_psi, tau_beta, tau_G, loss))

        results = pool.map(self.compute_grad_chunk, params)
        gamma_grad = np.concatenate([r for r in results])

        # g_star gradient
        chunk_size = K // mp.cpu_count()
        if chunk_size == 0:
            chunk_size = K
        params = []
        for k in range(0, K, chunk_size):
            k_start = k
            k_end = min(k + chunk_size, K)
            params.append((self.G_star, k_start, k_end, eps, tau_psi, tau_beta, tau_G, loss))

        results = pool.map(self.compute_grad_chunk, params)
        G_star_grad = np.concatenate([r for r in results])

        pool.close()
        pool.join()

        # d gradient
        d_grad = np.zeros_like(self.d)
        for k in range(K):
            orig = self.d[k]
            self.d[k] = orig + eps
            loss_d = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
            loss_eps = loss_d['log_likelihood'] + loss_d['psi_penalty'] + loss_d['beta_penalty']
            d_grad[k] = (loss_eps - loss) / eps
            self.d[k] = orig

        return alpha_grad, gamma_grad, G_star_grad, d_grad
