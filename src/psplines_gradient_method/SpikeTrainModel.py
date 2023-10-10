import numpy as np
import multiprocessing as mp
from src.psplines_gradient_method.general_functions import create_first_diff_matrix
from scipy.interpolate import BSpline
from scipy.sparse import csr_array, vstack

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
        self.dt = round(self.time[1] - self.time[0], 3)
        self.alpha_prime_multiply = None
        self.alpha_prime_add = None
        self.U_psi = None
        self.V = None
        self.knots = None
        self.knots_1 = None
        self.Omega_beta = None
        self.Omega_psi = None
        self.degree = None

    def initialize_for_time_warping(self, L, degree):

        # parameters
        K = self.Y.shape[0]
        P = self.time.shape[0] + 2
        Q = P  # will be equal to P now
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
        self.alpha_prime_add = np.zeros((K, Q))
        self.alpha_prime_add[:, 1] = 1
        self.alpha_prime_add = csr_array(self.alpha_prime_add)
        self.U_psi = csr_array(np.triu(np.ones((Q, Q))))
        self.Omega_beta = create_first_diff_matrix(P)
        self.Omega_beta = self.Omega_beta.T @ self.Omega_beta
        self.Omega_beta = csr_array(self.Omega_beta)
        self.Omega_psi = self.Omega_beta

        # variables
        np.random.seed(0)
        self.chi = np.random.rand(K, L)
        np.random.seed(0)
        self.gamma = np.random.rand(L, P)
        np.random.seed(0)
        self.d = np.random.rand(K, 1)
        np.random.seed(0)
        self.alpha = np.random.rand(K, Q)

        return self

    def log_obj_with_backtracking_line_search_and_time_warping(self, tau_psi, tau_beta,
                                                               alpha_factor=1e-2, gamma_factor=1e-2,
                                                               G_factor=1e-2, d_factor=1e-2,
                                                               alpha=0.3, max_iters=4):
        # define parameters
        K, Q = self.alpha.shape

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add
        psi = exp_alpha_c @ self.U_psi  # variable
        psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # variable, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1/np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
        # compute loss
        log_likelihood = np.sum(diagdJ_plus_GBetaB * self.Y - lambda_del_t)
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))
        loss = log_likelihood + psi_penalty + beta_penalty
        loss_0 = loss

        # smooth_gamma
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        dlogL_dgamma = beta * (G.T @ np.vstack([y_minus_lambda_del_t[i] @ b.transpose() for i, b in enumerate(B_sparse)]) -
                               2 * tau_beta * beta @ self.Omega_beta)
        while ct < max_iters:
            gamma_plus = self.gamma + learning_rate * dlogL_dgamma

            # set up variables to compute loss
            beta = np.exp(gamma_plus)
            GBeta = G @ beta
            GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])
            diagdJ_plus_GBetaB = self.d + GBetaBPsi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
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
        # B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # didnt change
        beta = np.exp(self.gamma)  # now fixed
        # exp_chi = np.exp(self.chi)  # didnt change
        # G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # didnt change
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
        # compute updated penalty
        beta_penalty = - tau_beta * np.sum(np.diag(beta @ self.Omega_beta @ beta.T))

        # smooth_alpha
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        knots_Bpsinminus1_1 = [(self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree-1)).transpose()).tocsc() for time in time_matrix]
        knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, self.time.shape[0]))]).tocsc() for b_deriv in knots_Bpsinminus1_1]
        dlogL_dalpha = np.vstack([self.degree * max(self.time) * ((GBeta[i] @ (knots_Bpsinminus1_1[i] - knots_Bpsinminus1_2[i])) @
                          ((psi_norm[i, 1] * exp_alpha_c[i, :, np.newaxis] * (self.U_psi - psi_norm[i]) @ self.V) * y_minus_lambda_del_t[i]).T) -
                                  2 * tau_psi * psi_norm[i, 1] * exp_alpha_c[i] * ((psi_norm[i] @ self.Omega_psi) @ (self.U_psi - psi_norm[i]).T)
                          for i in range(K)])
        # we multiply by max time here because in the likelihood we multiply by max time, so its the derivarive of a constant times a function of alpha.
        while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
            alpha_plus = self.alpha + learning_rate * dlogL_dalpha

            # set up variables to compute loss
            exp_alpha_c = (np.exp(alpha_plus) @ self.alpha_prime_multiply) + self.alpha_prime_add
            psi = exp_alpha_c @ self.U_psi  # variable
            psi_norm = (1 / (psi[:, (Q-1), np.newaxis])) * psi  # variable, called \psi' in the document
            time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
            B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]
            GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # variable
            diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
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
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # now fixed
        # beta = np.exp(self.gamma)  # now fixed
        # exp_chi = np.exp(self.chi)  # didnt change
        # G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # didnt change
        # GBeta = G @ beta  # didnt change
        GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
        # compute updated penalty
        psi_penalty = - tau_psi * np.sum(np.diag(psi_norm @ self.Omega_psi @ psi_norm.T))

        # smooth_chi
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        dlogL_dchi = np.vstack([((y_minus_lambda_del_t[i] @ b.transpose()) @ (G[i, :, np.newaxis] * (GBeta[i] - beta)).T) for i, b in enumerate(B_sparse)])
        while ct < max_iters:
            chi_plus = self.chi + learning_rate * dlogL_dchi

            # set up variables to compute loss
            exp_chi = np.exp(chi_plus)  # variable
            G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
            GBeta = G @ beta  # didnt change
            GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # variable
            diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
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
        # B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # now fixed
        # beta = np.exp(self.gamma)  # now fixed
        exp_chi = np.exp(self.chi)  # now fixed
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # now fixed
        GBeta = G @ beta  # now fixed
        GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # now fixed
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable

        # smooth_d
        ct = 0
        learning_rate = 1
        y_minus_lambda_del_t = self.Y - lambda_del_t
        dlogL_dd = np.sum(y_minus_lambda_del_t, axis=1)
        while ct < max_iters:
            d_plus = self.d + learning_rate * dlogL_dd

            # set up variables to compute loss
            diagdJ_plus_GBetaB = self.d + GBetaBPsi
            lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt
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
        K, Q = self.alpha.shape

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add
        psi = exp_alpha_c @ self.U_psi  # variable
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
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
        K, Q = self.alpha.shape

        # set up variables to compute loss
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + self.alpha_prime_add
        psi = exp_alpha_c @ self.U_psi  # variable
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        time_matrix = max(self.time) * (psi_norm @ self.V)  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        beta = np.exp(self.gamma)  # variable
        exp_chi = np.exp(self.chi)  # variable
        G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
        GBeta = G @ beta  # variable
        GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])  # variable
        diagdJ_plus_GBetaB = self.d + GBetaBPsi  # variable
        lambda_del_t = np.exp(diagdJ_plus_GBetaB) * self.dt  # variable
        # compute gradients
        y_minus_lambda_del_t = self.Y - lambda_del_t
        knots_Bpsinminus1_1 = [(self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for time in time_matrix]
        knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, self.time.shape[0]))]).tocsc() for b_deriv in knots_Bpsinminus1_1]

        dlogL_dgamma = beta * (G.T @ np.vstack([y_minus_lambda_del_t[i] @ b.transpose() for i, b in enumerate(B_sparse)]) -
                    2 * tau_beta * beta @ self.Omega_beta)


        dlogL_dalpha = np.vstack( [self.degree * max(self.time) * ((GBeta[i] @ (knots_Bpsinminus1_1[i] - knots_Bpsinminus1_2[i])) @
                    ((psi_norm[i, 1] * exp_alpha_c[i, :, np.newaxis] * (self.U_psi - psi_norm[i]) @ self.V) * y_minus_lambda_del_t[i]).T) -
             2 * tau_psi * psi_norm[i, 1] * exp_alpha_c[i] * ((psi_norm[i] @ self.Omega_psi) @ (self.U_psi - psi_norm[i]).T)
             for i in range(K)])

        dlogL_dchi = np.vstack([((y_minus_lambda_del_t[i] @ b.transpose()) @ (G[i, :, np.newaxis] * (GBeta[i] - beta)).T) for i, b in enumerate(B_sparse)])

        dlogL_dd = np.sum(y_minus_lambda_del_t, axis=1)

        return dlogL_dgamma, dlogL_dalpha, dlogL_dchi, dlogL_dd

    def compute_grad_chunk(self, name, variable, i, eps, tau_psi, tau_beta, loss):

        J = variable.shape[1]
        grad_chunk = np.zeros(J)

        print(f'Starting {name} gradient chunk at row: {i}')

        for j in range(J):
            orig = variable[i, j]
            variable[i, j] = orig + eps
            loss_result = self.compute_loss_time_warping(tau_psi, tau_beta)
            loss_eps = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
            grad_chunk[j] = (loss_eps - loss) / eps
            variable[i, j] = orig

        print(f'Completed {name} gradient chunk at row: {i}')

        return grad_chunk

    def compute_numerical_grad_time_warping_parallel(self, tau_psi, tau_beta):

        eps = 1e-4
        # define parameters
        K = self.Y.shape[0]
        L = self.gamma.shape[0]

        loss_result = self.compute_loss_time_warping(tau_psi, tau_beta)
        loss = loss_result['log_likelihood'] + loss_result['psi_penalty'] + loss_result['beta_penalty']
        pool = mp.Pool()

        # alpha gradient
        alpha_async_results = [pool.apply_async(self.compute_grad_chunk, args=('alpha', self.alpha, k, eps, tau_psi, tau_beta, loss)) for k in range(K)]

        # gamma gradient
        gamma_async_results = [pool.apply_async(self.compute_grad_chunk, args=('gamma', self.gamma, l, eps, tau_psi, tau_beta, loss)) for l in range(L)]

        # chi gradient
        chi_async_results = [pool.apply_async(self.compute_grad_chunk, args=('chi', self.chi, k, eps, tau_psi, tau_beta, loss)) for k in range(K)]

        # d gradient
        d_async_results = [pool.apply_async(self.compute_grad_chunk, args=('d', self.d, k, eps, tau_psi, tau_beta, loss)) for k in range(K)]

        pool.close()
        pool.join()

        alpha_grad = np.vstack([r.get() for r in alpha_async_results])
        gamma_grad = np.vstack([r.get() for r in gamma_async_results])
        chi_grad = np.vstack([r.get() for r in chi_async_results])
        d_grad = np.vstack([r.get() for r in d_async_results])

        return gamma_grad, alpha_grad, chi_grad, d_grad
