import numpy as np
from sklearn.isotonic import IsotonicRegression
import multiprocessing as mp
from src.psplines_gradient_method.general_functions import create_first_diff_matrix, create_masking_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bspline_functions, generate_bspline_matrix, bspline_deriv_multipliers


class SpikeTrainModel:
    def __init__(self, Y, time):
        # variables
        self.G = None
        self.G_star = None
        self.beta = None
        self.d = None
        self.psi = None

        # parameters
        self.Y = Y
        self.time = time
        self.mask_G = None
        self.J = None
        self.mask_beta = None
        self.I_beta_P = None
        self.I_beta_L = None
        self.mask_psi = None
        self.J_psi = None
        self.V = None
        self.B_func_n = None
        self.Omega_beta = None
        self.Omega_psi = None
        self.degree = None

    def initialize(self, L, degree):

        # parameters
        self.degree = degree
        self.B_func_n = generate_bspline_functions(self.time, self.degree)

        time_matrix = self.time[np.newaxis, :]
        # latent factor b-spline matrix. Re-using the self.V matrix. Coefficients would be from beta
        self.V = generate_bspline_matrix(self.B_func_n, time_matrix)
        K = self.Y.shape[0]
        P = len(self.B_func_n)
        self.J = np.ones_like(self.Y)
        self.Omega_beta = create_first_diff_matrix(P)
        self.Omega_beta = self.Omega_beta.T @ self.Omega_beta

        # variables
        np.random.seed(0)
        self.G = np.random.rand(K, L)
        np.random.seed(0)
        self.beta = np.maximum(np.random.rand(L, P), 0)
        np.random.seed(0)
        self.d = np.random.rand(K)

        return self

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
        self.mask_psi = np.hstack([np.eye(K)] * Q)
        self.J_psi = create_masking_matrix(Q, K).T
        self.Omega_beta = create_first_diff_matrix(P)
        self.Omega_beta = self.Omega_beta.T @ self.Omega_beta
        self.Omega_psi = create_first_diff_matrix(Q)
        self.Omega_psi = self.Omega_psi.T @ self.Omega_psi

        # variables
        np.random.seed(0)
        self.G_star = np.random.rand(K, K * L) * self.mask_G
        np.random.seed(0)
        self.beta = np.maximum(np.random.rand(L, P), 0)
        np.random.seed(0)
        self.d = np.random.rand(K)
        np.random.seed(0)
        iso_reg = IsotonicRegression()
        x = np.arange(Q)
        self.psi = np.maximum(np.array([iso_reg.fit_transform(x, row) for row in np.random.rand(K, Q)]), 0)

        return self

    def log_obj_with_backtracking_line_search(self, tau_beta, tau_G,
                                              beta_factor=1e-2, G_factor=1e-2, d_factor=1e-2,
                                              alpha=0.3, max_iters=4):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)

        # set up variables to compute loss
        B = self.V
        diagdJ = self.d[:, np.newaxis] * self.J  # variable
        diagdJ_plus_GBetaB = diagdJ + self.G @ self.beta @ B  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        beta_penalty = - tau_beta * np.sum(np.diag(self.beta @ self.Omega_beta @ self.beta.T))
        G_penalty = - tau_G * np.linalg.norm(self.G, ord=1)
        loss = log_likelihood + beta_penalty + G_penalty
        loss_0 = loss

        # smooth_beta
        ct = 0
        learning_rate = 1
        dlogL_dbeta = self.G.T @ (self.Y - lambda_del_t) @ B.T - 2 * tau_beta * self.beta @ self.Omega_beta
        while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
            beta_minus = self.beta + learning_rate * dlogL_dbeta
            beta_plus = np.maximum(beta_minus, 0)
            gen_grad_curr = (beta_plus - self.beta) / learning_rate

            # set up variables to compute loss
            diagdJ_plus_GBetaB = diagdJ + self.G @ beta_plus @ B
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            beta_penalty = - tau_beta * np.sum(np.diag(beta_plus @ self.Omega_beta @ beta_plus.T))
            loss_next = log_likelihood + beta_penalty + G_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if (loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dbeta * gen_grad_curr) +
                    alpha * learning_rate * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
                break
            learning_rate *= beta_factor
            ct += 1

        if ct < max_iters:
            ct_beta = ct
            smooth_beta = learning_rate
            loss = loss_next
            self.beta = beta_plus
        else:
            ct_beta = np.inf
            smooth_beta = 0
        loss_beta = loss

        # set up variables to compute loss in next round
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        betaB = self.beta @ B  # now fixed
        diagdJ_plus_GBetaB = diagdJ + self.G @ betaB  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        beta_penalty = - tau_beta * np.sum(np.diag(self.beta @ self.Omega_beta @ self.beta.T))

        # smooth_G
        ct = 0
        learning_rate = 1
        dlogL_dG = (self.Y - lambda_del_t) @ betaB.T
        while ct < max_iters:
            G_minus = self.G + learning_rate * dlogL_dG
            G_plus = np.maximum(np.abs(G_minus) - tau_G * learning_rate, 0) * np.sign(G_minus)
            gen_grad_curr = (G_plus - self.G) / learning_rate

            # set up variables to compute loss
            diagdJ_plus_GBetaB = diagdJ + G_plus @ betaB
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            G_penalty = - tau_G * np.linalg.norm(G_plus, ord=1)
            loss_next = log_likelihood + beta_penalty + G_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if (loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dG * gen_grad_curr) +
                    alpha * learning_rate * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
                break
            learning_rate *= G_factor
            ct += 1

        if ct < max_iters:
            ct_G = ct
            smooth_G = learning_rate
            loss = loss_next
            self.G = G_plus
        else:
            ct_G = np.inf
            smooth_G = 0
        loss_G = loss

        # set up variables to compute loss in next round
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        # betaB = self.beta @ B  # now fixed
        GBetaB = self.G @ betaB  # now fixed
        diagdJ_plus_GBetaB = diagdJ + GBetaB  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        G_penalty = - tau_G * np.linalg.norm(self.G, ord=1)

        # smooth_d
        ct = 0
        learning_rate = 1
        dlogL_dd = np.sum(self.Y - lambda_del_t, axis=1)
        while ct < max_iters:
            d_plus = self.d + learning_rate * dlogL_dd

            # set up variables to compute loss
            diagdJ_plus_GBetaB = d_plus[:, np.newaxis] * self.J + GBetaB
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            loss_next = log_likelihood + beta_penalty + G_penalty

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
            "dlogL_dbeta": dlogL_dbeta,
            "beta_loss_increase": loss_beta - loss_0,
            "smooth_beta": smooth_beta,
            "iters_beta": ct_beta,
            "dlogL_dG": dlogL_dG,
            "G_loss_increase": loss_G - loss_beta,
            "smooth_G": smooth_G,
            "iters_G": ct_G,
            "dlogL_dd": dlogL_dd,
            "d_loss_increase": loss_d - loss_G,
            "smooth_d": smooth_d,
            "iters_d": ct_d,
            "loss": loss,
            "beta_penalty": beta_penalty,
            "G_penalty": G_penalty
        }

        return result

    def log_obj_with_backtracking_line_search_and_time_warping(self, tau_psi, tau_beta, tau_G,
                                                               psi_factor=1e-2, beta_factor=1e-2,
                                                               G_factor=1e-2, d_factor=1e-2,
                                                               alpha=0.3, max_iters=4):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)
        K = self.Y.shape[0]
        P = len(self.B_func_n)
        Q = self.V.shape[0]
        iso_reg = IsotonicRegression()
        x = np.arange(Q)
        V_star = np.repeat(self.V, K, axis=0)

        # set up variables to compute loss
        time_matrix = self.psi @ self.V  # variable
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # variable
        diagdJ = self.d[:, np.newaxis] * self.J  # variable
        GStar_BetaStar = self.G_star @ np.kron(np.eye(K), self.beta)  # variable
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(self.psi @ self.Omega_psi @ self.psi.T))
        beta_penalty = - tau_beta * np.sum(np.diag(self.beta @ self.Omega_beta @ self.beta.T))
        G_penalty = - tau_G * np.linalg.norm(self.G_star, ord=1)
        loss = log_likelihood + psi_penalty + beta_penalty + G_penalty
        loss_0 = loss

        # smooth_psi
        # print('Optimizing psi')
        ct = 0
        learning_rate = 1
        B_psi_nminus1_1 = generate_bspline_matrix(self.B_func_nminus1, time_matrix)  # variable
        B_psi_nminus1_2 = np.zeros_like(B_psi_nminus1_1)  # variable
        for i in range(K):
            # Copy rows from A to B with a shift
            B_psi_nminus1_2[i * P:(((i + 1) * P) - 1), :] = B_psi_nminus1_1[((i * P) + 1):((i + 1) * P), :]
        y_minus_lambdadt_star = np.vstack([self.Y - lambda_del_t] * Q)
        # psi gradient
        dlogL_dpsi = self.degree * (((GStar_BetaStar @
                        ((np.vstack([self.knots_1] * K) * B_psi_nminus1_1) - (
                                np.vstack([self.knots_2] * K) * B_psi_nminus1_2)) @
                        (V_star * y_minus_lambdadt_star).T) * self.mask_psi) @
                      self.J_psi) - 2 * tau_psi * self.psi @ self.Omega_psi
        while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
            psi_minus = self.psi + learning_rate * dlogL_dpsi
            psi_plus = np.maximum(np.array([iso_reg.fit_transform(x, row) for row in psi_minus]), 0)
            gen_grad_curr = (psi_plus - self.psi) / learning_rate

            # set up variables to compute loss
            time_matrix = psi_plus @ self.V  # variable
            B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)
            diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            psi_penalty = - tau_psi * np.sum(np.diag(psi_plus @ self.Omega_psi @ psi_plus.T))
            loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if (loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dpsi * gen_grad_curr) +
                    alpha * learning_rate * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
                break
            learning_rate *= psi_factor
            ct += 1

        if ct < max_iters:
            ct_psi = ct
            smooth_psi = learning_rate
            loss = loss_next
            self.psi = psi_plus
        else:
            ct_psi = np.inf
            smooth_psi = 0
        loss_psi = loss

        # set up variables to compute loss in next round
        time_matrix = self.psi @ self.V  # now fixed
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        # GStar_BetaStar = self.G_star @ np.kron(np.eye(K), self.beta)  # didnt change
        diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        psi_penalty = - tau_psi * np.sum(np.diag(self.psi @ self.Omega_psi @ self.psi.T))

        # smooth_beta
        # print('Optimizing beta')
        ct = 0
        learning_rate = 1
        dlogL_dbeta = ((self.G_star @ self.I_beta_L).T @
                       (((self.Y - lambda_del_t) @ B_psi.T) * self.mask_beta) @
                       self.I_beta_P - 2 * tau_beta * self.beta @ self.Omega_beta)
        while ct < max_iters:
            beta_minus = self.beta + learning_rate * dlogL_dbeta
            beta_plus = np.maximum(beta_minus, 0)
            gen_grad_curr = (beta_plus - self.beta) / learning_rate

            # set up variables to compute loss
            GStar_BetaStar = self.G_star @ np.kron(np.eye(K), beta_plus)
            diagdJ_plus_GBetaB = diagdJ + GStar_BetaStar @ B_psi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            beta_penalty = - tau_beta * np.sum(np.diag(beta_plus @ self.Omega_beta @ beta_plus.T))
            loss_next = log_likelihood + psi_penalty + beta_penalty + G_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if (loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dbeta * gen_grad_curr) +
                    alpha * learning_rate * 0.5 * np.linalg.norm(gen_grad_curr, ord='fro') ** 2):
                break
            learning_rate *= beta_factor
            ct += 1

        if ct < max_iters:
            ct_beta = ct
            smooth_beta = learning_rate
            loss = loss_next
            self.beta = beta_plus
        else:
            ct_beta = np.inf
            smooth_beta = 0
        loss_beta = loss

        # set up variables to compute loss in next round
        # time_matrix = self.psi @ self.V  # now fixed
        # B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        betaStar_BPsi = np.kron(np.eye(K), self.beta) @ B_psi  # now fixed
        diagdJ_plus_GBetaB = diagdJ + self.G_star @ betaStar_BPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute updated penalty
        beta_penalty = - tau_beta * np.sum(np.diag(self.beta @ self.Omega_beta @ self.beta.T))

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
        # time_matrix = self.psi @ self.V  # now fixed
        # B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # now fixed
        # diagdJ = self.d[:, np.newaxis] * self.J  # didnt change
        # betaStar_BPsi = np.kron(np.eye(K), self.beta) @ B_psi  # now fixed
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
            "dlogL_dpsi": dlogL_dpsi,
            "psi_loss_increase": loss_psi - loss_0,
            "smooth_psi": smooth_psi,
            "iters_psi": ct_psi,
            "dlogL_dbeta": dlogL_dbeta,
            "beta_loss_increase": loss_beta - loss_psi,
            "smooth_beta": smooth_beta,
            "iters_beta": ct_beta,
            "dlogL_dG": dlogL_dG_star,
            "G_loss_increase": loss_G - loss_beta,
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

    def compute_loss(self, tau_beta, tau_G):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)

        # set up variables to compute loss
        B = self.V
        diagdJ_plus_GBetaB = self.d[:, np.newaxis] * self.J + self.G @ self.beta @ B
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        beta_penalty = - tau_beta * np.sum(np.diag(self.beta @ self.Omega_beta @ self.beta.T))
        G_penalty = - tau_G * np.linalg.norm(self.G, ord=1)
        loss = log_likelihood + beta_penalty + G_penalty

        result = {
            "loss": loss,
            "log_likelihood": log_likelihood,
            "beta_penalty": beta_penalty,
            "G_penalty": G_penalty
        }

        return result


    def compute_analytical_grad(self, tau_beta):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)

        # set up variables to compute gradients
        B = self.V
        diagdJ_plus_GBetaB = self.d[:, np.newaxis] * self.J + self.G @ self.beta @ B  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt  # variable
        # compute gradients
        dlogL_dbeta = self.G.T @ (self.Y - lambda_del_t) @ B.T - 2 * tau_beta * self.beta @ self.Omega_beta
        dlogL_dG = (self.Y - lambda_del_t) @ (self.beta @ B).T
        dlogL_dd = np.sum(self.Y - lambda_del_t, axis=1)

        return dlogL_dbeta, dlogL_dG, dlogL_dd


    def compute_numerical_grad(self, tau_beta, tau_G):

        eps = 1e-5
        # define parameters
        K = self.Y.shape[0]
        L = self.beta.shape[0]
        P = len(self.B_func_n)

        result = self.compute_loss(tau_beta, tau_G)
        loss = result['loss']
        log_likelihood = result['log_likelihood']
        beta_penalty = result['beta_penalty']

        # beta gradient
        beta_grad = np.zeros_like(self.beta)
        for l in range(L):
            for p in range(P):
                orig = self.beta[l, p]
                self.beta[l, p] = orig + eps
                loss_beta = self.compute_loss(tau_beta, tau_G)['loss']
                beta_grad[l, p] = (loss_beta - loss) / eps
                self.beta[l, p] = orig

        # g gradient
        G_grad = np.zeros_like(self.G)
        for k in range(K):
            for l in range(L):
                orig = self.G[k, l]
                self.G[k, l] = orig + eps
                result = self.compute_loss(tau_beta, tau_G)
                log_likelihood_G = result['log_likelihood']
                beta_penalty_G = result['beta_penalty']
                G_grad[k, l] = ((log_likelihood_G+beta_penalty_G) - (log_likelihood+beta_penalty)) / eps
                self.G[k, l] = orig

        # d gradient
        d_grad = np.zeros_like(self.d)
        for k in range(K):
            orig = self.d[k]
            self.d[k] = orig + eps
            loss_d = self.compute_loss(tau_beta, tau_G)['loss']
            d_grad[k] = (loss_d - loss) / eps
            self.d[k] = orig

        return beta_grad, G_grad, d_grad


    def compute_loss_time_warping(self, tau_psi, tau_beta, tau_G):
        # define parameters
        dt = round(self.time[1] - self.time[0], 3)
        K = self.Y.shape[0]

        # set up variables to compute loss
        time_matrix = self.psi @ self.V  # variable
        # time_matrix = np.repeat(self.time[np.newaxis, :], K, axis=0); tau_psi = 0
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)  # variable
        diagdJ_plus_GBetaB = self.d[:, np.newaxis] * self.J + self.G_star @ np.kron(np.eye(K), self.beta) @ B_psi
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(self.psi @ self.Omega_psi @ self.psi.T))
        beta_penalty = - tau_beta * np.sum(np.diag(self.beta @ self.Omega_beta @ self.beta.T))
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
        iso_reg = IsotonicRegression()
        x = np.arange(Q)
        V_star = np.repeat(self.V, K, axis=0)

        # set up variables to compute loss
        time_matrix = self.psi @ self.V  # variable
        # time_matrix = np.repeat(self.time[np.newaxis, :], K, axis=0); tau_psi = 0
        B_psi = generate_bspline_matrix(self.B_func_n, time_matrix)
        GStar_BetaStar = self.G_star @ np.kron(np.eye(K), self.beta)  # variable
        diagdJ_plus_GBetaB = self.d[:, np.newaxis] * self.J + GStar_BetaStar @ B_psi
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
        # compute gradients
        B_psi_nminus1_1 = generate_bspline_matrix(self.B_func_nminus1, time_matrix)  # variable
        B_psi_nminus1_2 = np.zeros_like(B_psi_nminus1_1)  # variable
        for i in range(K):
            # Copy rows from A to B with a shift
            B_psi_nminus1_2[i * P:(((i + 1) * P) - 1), :] = B_psi_nminus1_1[((i * P) + 1):((i + 1) * P), :]
        y_minus_lambdadt_star = np.vstack([self.Y - lambda_del_t] * Q)
        # psi gradient
        dlogL_dpsi = self.degree * (((GStar_BetaStar @
                        ((np.vstack([self.knots_1] * K) * B_psi_nminus1_1) - (
                                np.vstack([self.knots_2] * K) * B_psi_nminus1_2)) @
                        (V_star * y_minus_lambdadt_star).T) * self.mask_psi) @
                      self.J_psi) - 2 * tau_psi * self.psi @ self.Omega_psi

        dlogL_dbeta = ((self.G_star @ self.I_beta_L).T @
                       (((self.Y - lambda_del_t) @ B_psi.T) * self.mask_beta) @
                       self.I_beta_P - 2 * tau_beta * self.beta @ self.Omega_beta)

        dlogL_dG_star = ((self.Y - lambda_del_t) @ (np.kron(np.eye(K), self.beta) @ B_psi).T) * self.mask_G

        dlogL_dd = np.sum(self.Y - lambda_del_t, axis=1)

        return dlogL_dpsi, dlogL_dbeta, dlogL_dG_star, dlogL_dd


    def compute_grad_chunk(self, params):
        variable, i_start, i_end, eps, tau_psi, tau_beta, tau_G, loss = params
        J = variable.shape[1]
        J2 = 0
        grad_chunk = np.zeros((i_end - i_start, J))

        if variable.shape == self.psi.shape:
            name = 'psi'
        elif variable.shape == self.beta.shape:
            name = 'beta'
        elif variable.shape == self.G_star.shape:
            name = 'G_star'
            J = variable.shape[1]//variable.shape[0]
            J2 = J

        print(f'Starting {name} gradient chunk. i_start: {i_start}, i_end: {i_end}')

        for i in range(i_start, i_end):
            for j in (np.arange(J) + (i * J2)):
                orig = variable[i, j]
                variable[i, j] = orig + eps
                loss_result = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
                loss_eps = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
                grad_chunk[i-i_start, j] = (loss_eps - loss) / eps
                variable[i, j] = orig

        print(f'Completed {name} gradient chunk. i_start: {i_start}, i_end: {i_end}')

        return grad_chunk


    def compute_numerical_grad_time_warping_parallel(self, tau_psi, tau_beta, tau_G):

        eps = 1e-5
        # define parameters
        K = self.Y.shape[0]
        L = self.beta.shape[0]

        loss_result = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
        loss = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
        pool = mp.Pool()

        # psi gradient
        chunk_size = K // mp.cpu_count()
        if chunk_size == 0:
            chunk_size = K
        params = []
        for k in range(0, K, chunk_size):
            k_start = k
            k_end = min(k + chunk_size, K)
            params.append((self.psi, k_start, k_end, eps, tau_psi, tau_beta, tau_G, loss))

        results = pool.map(self.compute_grad_chunk, params)
        psi_grad = np.concatenate([r for r in results])

        # beta gradient
        chunk_size = L // mp.cpu_count()
        if chunk_size == 0:
            chunk_size = L
        params = []
        for l in range(0, L, chunk_size):
            l_start = l
            l_end = min(l + chunk_size, L)
            params.append((self.beta, l_start, l_end, eps, tau_psi, tau_beta, tau_G, loss))

        results = pool.map(self.compute_grad_chunk, params)
        beta_grad = np.concatenate([r for r in results])

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

        return psi_grad, beta_grad, G_star_grad, d_grad


    def compute_numerical_grad_time_warping(self, tau_psi, tau_beta, tau_G):

        eps = 1e-5
        # define parameters
        K = self.Y.shape[0]
        L = self.beta.shape[0]
        P = len(self.B_func_n)
        Q = self.V.shape[0]

        result = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
        loss = result['loss']
        log_likelihood = result['log_likelihood']
        psi_penalty = result['psi_penalty']
        beta_penalty = result['beta_penalty']

        # psi gradient
        psi_grad = np.zeros_like(self.psi)
        for k in range(K):
            for q in range(Q):
                orig = self.psi[k, q]
                self.psi[k, q] = orig + eps
                loss_psi = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)['loss']
                psi_grad[k, q] = (loss_psi - loss) / eps
                self.psi[k, q] = orig

        # beta gradient
        beta_grad = np.zeros_like(self.beta)
        for l in range(L):
            for p in range(P):
                orig = self.beta[l, p]
                self.beta[l, p] = orig + eps
                loss_beta = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)['loss']
                beta_grad[l, p] = (loss_beta - loss) / eps
                self.beta[l, p] = orig

        # g_star gradient
        G_star_grad = np.zeros_like(self.G_star)
        for k in range(K):
            for l in (np.arange(L) + (k * L)):
                orig = self.G_star[k, l]
                self.G_star[k, l] = orig + eps
                result = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)
                log_likelihood_G = result['log_likelihood']
                psi_penalty_G = result['psi_penalty']
                beta_penalty_G = result['beta_penalty']
                G_star_grad[k, l] = ((log_likelihood_G+psi_penalty_G+beta_penalty_G) - (log_likelihood+psi_penalty+beta_penalty)) / eps
                self.G_star[k, l] = orig

        # d gradient
        d_grad = np.zeros_like(self.d)
        for k in range(K):
            orig = self.d[k]
            self.d[k] = orig + eps
            loss_d = self.compute_loss_time_warping(tau_psi, tau_beta, tau_G)['loss']
            d_grad[k] = (loss_d - loss) / eps
            self.d[k] = orig

        return psi_grad, beta_grad, G_star_grad, d_grad
