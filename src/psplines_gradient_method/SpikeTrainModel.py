import numpy as np
import multiprocessing as mp
from src.psplines_gradient_method.general_functions import create_first_diff_matrix
from scipy.interpolate import BSpline
from scipy.sparse import csr_array, vstack, hstack

class SpikeTrainModel:
    def __init__(self, Y, time):
        # variables
        self.chi = None
        self.gamma = None
        self.d = None
        self.alpha = None
        self.zeta = None

        # parameters
        self.Y = Y
        self.time = time
        self.trials = int(self.Y.shape[1]/self.time.shape[0])
        self.dt = round(self.time[1] - self.time[0], 3)
        self.alpha_prime_multiply = None
        self.alpha_prime_add = None
        self.U_ones = None
        self.V = None
        self.knots = None
        self.knots_1 = None
        self.Omega_beta_B = None
        self.Omega_psi_B = None
        self.degree = None

    def initialize_for_time_warping(self, L, degree):

        # parameters
        K = self.Y.shape[0]
        T = self.time.shape[0]
        P = T + 2
        Q = P  # will be equal to P now
        R = self.trials
        self.degree = degree
        self.knots = np.concatenate([np.repeat(self.time[0], degree), self.time, np.repeat(self.time[-1], degree)])
        self.knots[-1] = self.knots[-1] + self.dt
        self.knots_1 = (1 / (self.knots[degree:(P + degree)] - self.knots[0:P]))[:, np.newaxis]
        self.knots_1[0, 0] = 0

        # time warping b-spline matrix. Coefficients would be from psi
        self.V = BSpline.design_matrix(self.time, self.knots, self.degree).transpose()
        self.alpha_prime_multiply = np.eye(Q)
        self.alpha_prime_multiply[0, 0] = 0
        self.alpha_prime_multiply[1, 1] = 0
        self.alpha_prime_multiply = csr_array(self.alpha_prime_multiply)
        self.alpha_prime_add = np.zeros((1, Q))
        self.alpha_prime_add[:, 1] = 1
        self.U_ones = np.triu(np.ones((Q, Q)))
        self.Omega_beta_B = csr_array(create_first_diff_matrix(T)) @ self.V.T
        self.Omega_beta_B = self.Omega_beta_B.T @ self.Omega_beta_B
        self.Omega_psi_B = self.Omega_beta_B

        # variables
        np.random.seed(0)
        self.chi = np.random.rand(K, L)
        np.random.seed(0)
        self.gamma = np.random.rand(L, P)
        np.random.seed(0)
        self.d = np.random.rand(K, 1)
        self.alpha = np.zeros((K, Q))
        self.zeta = np.zeros((R, Q))

        return self

    def log_obj_with_backtracking_line_search_and_time_warping(self, tau_psi, tau_beta,
                                                               alpha_factor=1e-2, gamma_factor=1e-2,
                                                               G_factor=1e-2, d_factor=1e-2,
                                                               alpha=0.3, max_iters=4):
        # define parameters
        K, L = self.chi.shape
        R, Q = self.zeta.shape
        T = self.time.shape[0]

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)
        psi = exp_alpha_c @ self.U_ones  # variable
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # variable, called \psi' in the document
        exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)
        kappa = exp_zeta_c @ self.U_ones  # variable
        kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
        time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1/np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
        kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        beta_penalty = - tau_beta * np.sum((beta @ self.Omega_beta_B) * beta)
        loss = log_likelihood + psi_penalty + kappa_penalty + beta_penalty
        loss_0 = loss

        # smooth_gamma
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        dlogL_dgamma = beta * (G.T @ np.vstack([y_minus_lambda_del_t[k] @ b.transpose() for k, b in enumerate(B_sparse)]) -
                               2 * tau_beta * beta @ self.Omega_beta_B)
        while ct < max_iters:
            gamma_plus = self.gamma + learning_rate * dlogL_dgamma

            # set up variables to compute loss
            beta = np.exp(gamma_plus)
            GBeta = G @ beta
            GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])
            diagdJ_plus_GBetaB = self.d + GBetaBPsi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            beta_penalty = - tau_beta * np.sum((beta @ self.Omega_beta_B) * beta)
            loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_penalty

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
        # exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)  # didnt change
        # psi = exp_alpha_c @ self.U_ones  # didnt change
        # psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # didnt change, called \psi' in the document
        # exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)  # didnt change
        # kappa = exp_zeta_c @ self.U_ones  # didnt change
        # kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # didnt change, called \kappa' in the document
        # time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # didnt change
        # B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # didnt change
        beta = np.exp(self.gamma)  # now fixed
        # exp_chi = np.exp(self.chi)  # didnt change
        # G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # didnt change
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
        # compute updated penalty
        beta_penalty = - tau_beta * np.sum((beta @ self.Omega_beta_B) * beta)

        # smooth_alpha
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        knots_Bpsinminus1_1 = [(self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for time in time_matrix]
        knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R*T))]).tocsc() for b_deriv in knots_Bpsinminus1_1]
        GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
        GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
        psi_norm_Omega = psi_norm @ self.Omega_psi_B
        dlogL_dalpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c * np.vstack(
            [0.5 * max(self.time) * self.degree * np.sum(GBetaBPsiDerivXyLambda * ((self.U_ones[q] - psi_norm) @ hstack([self.V]*R)), axis=1) -
             2 * tau_psi * np.sum(psi_norm_Omega * (self.U_ones[q] - psi_norm), axis=1)
             for q in range(Q)]).T
        # we multiply by max time here because in the likelihood we multiply by max time, so its the derivarive of a constant times a function of alpha.
        while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
            alpha_plus = self.alpha + learning_rate * dlogL_dalpha

            # set up variables to compute loss
            exp_alpha_c = (np.exp(alpha_plus) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)
            psi = exp_alpha_c @ self.U_ones  # variable
            psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
            time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
            B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]
            GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
            diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
            loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_penalty

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
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)  # now fixed
        psi = exp_alpha_c @ self.U_ones  # now fixed
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        # exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)  # didnt change
        # kappa = exp_zeta_c @ self.U_ones  # didnt change
        # kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # didnt change, called \kappa' in the document
        time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        # beta = np.exp(self.gamma)  # now fixed
        # exp_chi = np.exp(self.chi)  # didnt change
        # G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # didnt change
        # GBeta = G @ beta  # didnt change
        GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
        # compute updated penalty
        psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)

        # smooth_zeta
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        knots_Bpsinminus1_1 = [(self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for time in time_matrix]
        knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R*T))]).tocsc() for b_deriv in knots_Bpsinminus1_1]
        GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
        GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
        kappa_norm_Omega = kappa_norm @ self.Omega_psi_B
        dlogL_dzeta = kappa_norm[:, 1, np.newaxis] * exp_zeta_c * np.vstack([0.5 * max(self.time) * self.degree *
             np.sum(np.sum(GBetaBPsiDerivXyLambda * ((self.U_ones[q] - kappa_norm) @ self.V).flatten(), axis=0).reshape((R, T)), axis=1) -
             2 * tau_psi * np.sum(kappa_norm_Omega * (self.U_ones[q] - kappa_norm), axis=1)
             for q in range(Q)]).T
        # we multiply by max time here because in the likelihood we multiply by max time, so its the derivarive of a constant times a function of alpha.
        while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
            zeta_plus = self.zeta + learning_rate * dlogL_dzeta

            # set up variables to compute loss
            exp_zeta_c = (np.exp(zeta_plus) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)
            kappa = exp_zeta_c @ self.U_ones  # variable
            kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
            time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
            B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]
            GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
            diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
            loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_penalty

            # Armijo condition, using Frobenius norm for matrices, but for maximization
            if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dzeta, ord='fro') ** 2:
                break
            learning_rate *= alpha_factor
            ct += 1

        if ct < max_iters:
            ct_zeta = ct
            smooth_zeta = learning_rate
            loss = loss_next
            self.zeta = zeta_plus
        else:
            ct_zeta = np.inf
            smooth_zeta = 0
        loss_zeta = loss

        # set up variables to compute loss in next round
        # exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)  # now fixed
        # psi = exp_alpha_c @ self.U_ones  # now fixed
        # psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)  # now fixed
        kappa = exp_zeta_c @ self.U_ones  # now fixed
        kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # now fixed, called \kappa' in the document
        time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # now fixed
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # now fixed
        # beta = np.exp(self.gamma)  # now fixed
        # exp_chi = np.exp(self.chi)  # didnt change
        # G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # didnt change
        # GBeta = G @ beta  # didnt change
        GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
        # compute updated penalty
        kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)

        # smooth_chi
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        dlogL_dchi = np.vstack([y_minus_lambda_del_t[k] @ (G[k, :, np.newaxis] * (np.eye(L) - G[k]) @ beta @ b).T for k, b in enumerate(B_sparse)])
        while ct < max_iters:
            chi_plus = self.chi + learning_rate * dlogL_dchi

            # set up variables to compute loss
            exp_chi = np.exp(chi_plus)  # variable
            G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
            GBeta = G @ beta  # didnt change
            GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
            diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_penalty

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
        # exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)  # now fixed
        # psi = exp_alpha_c @ self.U_ones  # now fixed
        # psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        # exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)  # now fixed
        # kappa = exp_zeta_c @ self.U_ones  # now fixed
        # kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # now fixed, called \kappa' in the document
        # time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # now fixed
        # B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # now fixed
        # beta = np.exp(self.gamma)  # now fixed
        exp_chi = np.exp(self.chi)  # now fixed
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # now fixed
        GBeta = G @ beta  # now fixed
        GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # now fixed
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable

        # smooth_d
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        dlogL_dd = np.sum(y_minus_lambda_del_t, axis=1)[:, np.newaxis]
        while ct < max_iters:
            d_plus = self.d + learning_rate * dlogL_dd

            # set up variables to compute loss
            diagdJ_plus_GBetaB = d_plus + GBetaBPsi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
            # compute loss
            log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
            loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_penalty

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
            "dlogL_dzeta": dlogL_dzeta,
            "zeta_loss_increase": loss_zeta - loss_alpha,
            "smooth_zeta": smooth_zeta,
            "iters_zeta": ct_zeta,
            "dlogL_dchi": dlogL_dchi,
            "chi_loss_increase": loss_chi - loss_zeta,
            "smooth_chi": smooth_chi,
            "iters_chi": ct_chi,
            "dlogL_dd": dlogL_dd,
            "d_loss_increase": loss_d - loss_chi,
            "smooth_d": smooth_d,
            "iters_d": ct_d,
            "likelihood": loss,
            "beta_penalty": beta_penalty,
            "psi_penalty": psi_penalty,
            "kappa_penalty": kappa_penalty
        }

        return result

    def compute_loss_time_warping(self, tau_psi, tau_beta):

        # define parameters
        K, Q = self.alpha.shape
        R = self.trials

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)
        psi = exp_alpha_c @ self.U_ones  # variable
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)
        kappa = exp_zeta_c @ self.U_ones  # variable
        kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
        time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
        kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        beta_penalty = - tau_beta * np.sum((beta @ self.Omega_beta_B) * beta)
        loss = log_likelihood + psi_penalty + kappa_penalty + beta_penalty

        result = {
            "likelihood": loss,
            "psi_penalty": psi_penalty,
            "beta_penalty": beta_penalty,
            "kappa_penalty": kappa_penalty
        }
        return result

    def compute_analytical_grad_time_warping(self, tau_psi, tau_beta):

        # define parameters
        K, L = self.chi.shape
        R, Q = self.zeta.shape
        T = self.time.shape[0]

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)
        psi = exp_alpha_c @ self.U_ones  # variable
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)
        kappa = exp_zeta_c @ self.U_ones  # variable
        kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
        time_matrix = 0.5 * max(self.time) * np.hstack([(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
        # compute gradients
        y_minus_lambda_del_t = self.Y - lambda_del_t
        knots_Bpsinminus1_1 = [(self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for time in time_matrix]
        knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R * T))]).tocsc() for b_deriv in knots_Bpsinminus1_1]
        GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
        GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
        psi_norm_Omega = psi_norm @ self.Omega_psi_B
        kappa_norm_Omega = kappa_norm @ self.Omega_psi_B

        dlogL_dgamma = beta * (G.T @ np.vstack([y_minus_lambda_del_t[k] @ b.transpose() for k, b in enumerate(B_sparse)]) -
                               2 * tau_beta * beta @ self.Omega_beta_B)

        dlogL_dalpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c * np.vstack([0.5 * max(self.time) * self.degree * np.sum(
                GBetaBPsiDerivXyLambda * ((self.U_ones[q] - psi_norm) @ hstack([self.V] * R)), axis=1) -
                2 * tau_psi * np.sum(psi_norm_Omega * (self.U_ones[q] - psi_norm), axis=1)
                for q in range(Q)]).T

        dlogL_dzeta = kappa_norm[:, 1, np.newaxis] * exp_zeta_c * np.vstack([0.5 * max(self.time) * self.degree *
                np.sum(np.sum(GBetaBPsiDerivXyLambda * ((self.U_ones[q] - kappa_norm) @ self.V).flatten(), axis=0).reshape((R, T)), axis=1) -
                2 * tau_psi * np.sum(kappa_norm_Omega * (self.U_ones[q] - kappa_norm), axis=1)
                for q in range(Q)]).T

        dlogL_dchi = np.vstack([y_minus_lambda_del_t[k] @ (G[k, :, np.newaxis] * (np.eye(L) - G[k]) @ beta @ b).T for k, b in enumerate(B_sparse)])

        dlogL_dd = np.sum(y_minus_lambda_del_t, axis=1)[:, np.newaxis]

        return dlogL_dgamma, dlogL_dalpha, dlogL_dzeta, dlogL_dchi, dlogL_dd

    def compute_grad_chunk(self, name, variable, i, eps, tau_psi, tau_beta, loss):

        J = variable.shape[1]
        grad_chunk = np.zeros(J)

        print(f'Starting {name} gradient chunk at row: {i}')

        for j in range(J):
            orig = variable[i, j]
            variable[i, j] = orig + eps
            loss_result = self.compute_loss_time_warping(tau_psi, tau_beta)
            loss_eps = loss_result['likelihood']
            grad_chunk[j] = (loss_eps - loss) / eps
            variable[i, j] = orig

        print(f'Completed {name} gradient chunk at row: {i}')

        return grad_chunk

    def compute_numerical_grad_time_warping_parallel(self, tau_psi, tau_beta):

        eps = 1e-4
        # define parameters
        K, L = self.chi.shape
        R = self.trials

        loss_result = self.compute_loss_time_warping(tau_psi, tau_beta)
        loss = loss_result['likelihood']
        pool = mp.Pool()

        # alpha gradient
        alpha_async_results = [pool.apply_async(self.compute_grad_chunk, args=('alpha', self.alpha, k, eps, tau_psi, tau_beta, loss)) for k in range(K)]

        # zeta gradient
        zeta_async_results = [pool.apply_async(self.compute_grad_chunk, args=('zeta', self.zeta, r, eps, tau_psi, tau_beta, loss)) for r in range(R)]

        # gamma gradient
        gamma_async_results = [pool.apply_async(self.compute_grad_chunk, args=('gamma', self.gamma, l, eps, tau_psi, tau_beta, loss)) for l in range(L)]

        # chi gradient
        chi_async_results = [pool.apply_async(self.compute_grad_chunk, args=('chi', self.chi, k, eps, tau_psi, tau_beta, loss)) for k in range(K)]

        # d gradient
        d_async_results = [pool.apply_async(self.compute_grad_chunk, args=('d', self.d, k, eps, tau_psi, tau_beta, loss)) for k in range(K)]

        pool.close()
        pool.join()

        alpha_grad = np.vstack([r.get() for r in alpha_async_results])
        zeta_grad = np.vstack([r.get() for r in zeta_async_results])
        gamma_grad = np.vstack([r.get() for r in gamma_async_results])
        chi_grad = np.vstack([r.get() for r in chi_async_results])
        d_grad = np.vstack([r.get() for r in d_async_results])

        return gamma_grad, alpha_grad, zeta_grad, chi_grad, d_grad
