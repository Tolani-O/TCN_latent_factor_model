import numpy as np
import multiprocessing as mp
from src.psplines_gradient_method.general_functions import create_first_diff_matrix, create_second_diff_matrix
from scipy.interpolate import BSpline
from scipy.sparse import csr_array, vstack, hstack


class SpikeTrainModel:
    def __init__(self, Y, time):
        # variables
        self.chi = None
        self.gamma = None
        self.alpha = None
        self.zeta = None
        self.d1 = None
        self.d2 = None

        # parameters
        self.Y = Y
        self.time = time
        self.trials = int(self.Y.shape[1] / self.time.shape[0])
        self.dt = round(self.time[1] - self.time[0], 3)
        self.alpha_prime_multiply = None
        self.alpha_prime_add = None
        self.U_ones = None
        self.V = None
        self.knots = None
        self.knots_1 = None
        self.BDelta1TDelta1BT = None
        self.Delta2BT = None
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
        Delta1BT = csr_array(create_first_diff_matrix(T)) @ self.V.T
        self.BDelta1TDelta1BT = Delta1BT.T @ Delta1BT
        Delta2BT = csr_array(create_second_diff_matrix(T)) @ self.V.T
        self.BDelta2TDelta2BT = Delta2BT.T @ Delta2BT
        self.Omega_psi_B = self.BDelta2TDelta2BT

        # variables
        # np.random.seed(0)
        self.chi = np.random.rand(K, L)
        self.chi[:, 0] = 0
        # np.random.seed(0)
        self.gamma = np.random.rand(L, P)
        self.alpha = np.zeros((K, Q))
        self.zeta = np.zeros((R, Q))
        self.d1 = np.zeros((L, 1))
        self.d2 = np.zeros((L, 1))

        return self

    def log_obj_with_backtracking_line_search_and_time_warping(self, tau_psi, tau_beta, tau_s, beta_first=1,
                                                               time_warping=False,
                                                               alpha_factor=1e-2, gamma_factor=1e-2,
                                                               G_factor=1e-2, d_factor=1e-2,
                                                               alpha=0.3, max_iters=4):
        # define parameters
        K, L = self.chi.shape
        T = self.time.shape[0]

        # set up variables to compute loss
        objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s, time_warping)
        # exp_alpha_c = objects["exp_alpha_c"]
        # exp_zeta_c = objects["exp_zeta_c"]
        # kappa_norm = objects["kappa_norm"]
        # psi_norm = objects["psi_norm"]
        # time_matrix = objects["time_matrix"]
        B_sparse = objects["B_sparse"]
        psi_penalty = objects["psi_penalty"]
        kappa_penalty = objects["kappa_penalty"]
        beta_s1_penalty = objects["beta_s1_penalty"]
        s1_penalty = objects["s1_penalty"]
        s1_norm = objects["s1_norm"]
        beta_s2_penalty = objects["beta_s2_penalty"]
        s2_penalty = objects["s2_penalty"]
        s2_norm = objects["s2_norm"]
        maxes = objects["maxes"]
        sum_exps_chi = objects["sum_exps_chi"]
        sum_exps_chi_plus_gamma_B = objects["sum_exps_chi_plus_gamma_B"]
        max_gamma = objects["max_gamma"]
        beta_minus_max = objects["beta_minus_max"]
        loss = objects["loss"]
        loss_0 = loss

        if beta_first:
            # smooth_gamma
            ct = 0
            learning_rate = 1
            exp_chi = np.vstack([np.exp(self.chi[k] - maxes[k]) for k in range(K)])  # variable
            likelihood_component = exp_chi.T @ np.vstack([(1/(sum_exps_chi_plus_gamma_B[k]) * self.Y[k] - 1/sum_exps_chi[k] * self.dt) @ b.transpose() for k, b in enumerate(B_sparse)])
            s1_component = s1_norm * beta_minus_max @ self.BDelta1TDelta1BT
            s2_component = s2_norm * beta_minus_max @ self.BDelta2TDelta2BT
            dlogL_dgamma = beta_minus_max * np.exp(max_gamma) * (likelihood_component - 2 * tau_beta * np.exp(max_gamma) * (s1_component + s2_component))
            while ct < max_iters:
                gamma_plus = self.gamma + learning_rate * dlogL_dgamma

                # set up variables to compute loss
                maxes = [np.max(self.chi[k][:, np.newaxis] + gamma_plus) for k in range(K)]
                sum_exps_chi = [np.sum(np.exp(self.chi[k] - maxes[k])) for k in range(K)]
                sum_exps_chi_plus_gamma_B = [np.sum(np.exp(self.chi[k][:, np.newaxis] + gamma_plus - maxes[k]), axis=0)[np.newaxis, :] @ b for k, b in enumerate(B_sparse)]
                log_likelihood = np.sum(np.vstack([(np.log(sum_exps_chi_plus_gamma_B[k]) - np.log(sum_exps_chi[k])) * self.Y[k] -
                               (1 / sum_exps_chi[k] * sum_exps_chi_plus_gamma_B[k]) * self.dt for k in range(K)]))
                max_gamma = np.max(gamma_plus)
                beta_minus_max = np.exp(gamma_plus - max_gamma)
                beta_s1_penalty = - tau_beta * np.exp(2 * max_gamma) * (s1_norm.T @ np.sum((beta_minus_max @ self.BDelta1TDelta1BT) * beta_minus_max, axis=1)).squeeze()
                beta_s2_penalty = - tau_beta * np.exp(2 * max_gamma) * (s2_norm.T @ np.sum((beta_minus_max @ self.BDelta2TDelta2BT) * beta_minus_max, axis=1)).squeeze()
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty
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
            maxes = [np.max(self.chi[k][:, np.newaxis] + self.gamma) for k in range(K)]
            sum_exps_chi = [np.sum(np.exp(self.chi[k] - maxes[k])) for k in range(K)]
            sum_exps_chi_plus_gamma_B = [np.sum(np.exp(self.chi[k][:, np.newaxis] + self.gamma - maxes[k]), axis=0)[np.newaxis, :] @ b for k, b in enumerate(B_sparse)]
            log_likelihood = np.sum(np.vstack([(np.log(sum_exps_chi_plus_gamma_B[k]) - np.log(sum_exps_chi[k])) * self.Y[k] -
                           (1 / sum_exps_chi[k] * sum_exps_chi_plus_gamma_B[k]) * self.dt for k in range(K)]))
            max_gamma = np.max(self.gamma)
            beta_minus_max = np.exp(self.gamma - max_gamma)

            # smooth_d1
            ct = 0
            learning_rate = 1
            diagBetaDeltaBeta = np.sum((beta_minus_max @ self.BDelta1TDelta1BT) * beta_minus_max, axis=1)[:, np.newaxis]
            dlogL_dd1 = s1_norm * (s1_norm.T - np.eye(L)) @ (tau_beta * np.exp(2 * max_gamma) * diagBetaDeltaBeta +
                                                             2 * tau_s * (s1_norm - 1 / L))
            while ct < max_iters:
                d1_plus = self.d1 + learning_rate * dlogL_dd1

                # set up variables to compute loss
                s1 = np.exp(d1_plus)
                s1[0, :] = 1
                s1_norm = (1 / np.sum(s1)) * s1
                beta_s1_penalty = - tau_beta * np.exp(2 * max_gamma) * (s1_norm.T @ diagBetaDeltaBeta).squeeze()
                s1_norm_minus_linv = s1_norm - 1 / L
                s1_penalty = - tau_s * (s1_norm_minus_linv.T @ s1_norm_minus_linv).squeeze()
                # compute loss
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty

                # Armijo condition, using l2 norm, but for maximization
                if loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dd1 * dlogL_dd1):
                    break
                learning_rate *= d_factor
                ct += 1

            if ct < max_iters:
                ct_d1 = ct
                smooth_d1 = learning_rate
                loss = loss_next
                self.d1 = d1_plus
            else:
                ct_d1 = np.inf
                smooth_d1 = 0
            loss_d1 = loss

            # set up variables to compute loss in next round
            s1 = np.exp(self.d1)
            s1[0, :] = 1
            s1_norm = (1 / np.sum(s1)) * s1
            beta_s1_penalty = - tau_beta * np.exp(2 * max_gamma) * (s1_norm.T @ diagBetaDeltaBeta).squeeze()
            s1_norm_minus_linv = s1_norm - 1 / L
            s1_penalty = - tau_s * (s1_norm_minus_linv.T @ s1_norm_minus_linv).squeeze()

            # smooth_d2
            ct = 0
            learning_rate = 1
            diagBetaDeltaBeta = np.sum((beta_minus_max @ self.BDelta2TDelta2BT) * beta_minus_max, axis=1)[:, np.newaxis]
            dlogL_dd2 = s2_norm * (s2_norm.T - np.eye(L)) @ (tau_beta * np.exp(2 * max_gamma) * diagBetaDeltaBeta +
                                                             2 * tau_s * (s2_norm - 1 / L))
            while ct < max_iters:
                d2_plus = self.d2 + learning_rate * dlogL_dd2

                # set up variables to compute loss
                s2 = np.exp(d2_plus)
                s2[0, :] = 1
                s2_norm = (1 / np.sum(s2)) * s2
                beta_s2_penalty = - tau_beta * np.exp(2 * max_gamma) * (s2_norm.T @ diagBetaDeltaBeta).squeeze()
                s2_norm_minus_linv = s2_norm - 1 / L
                s2_penalty = - tau_s * (s2_norm_minus_linv.T @ s2_norm_minus_linv).squeeze()
                # compute loss
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty

                # Armijo condition, using l2 norm, but for maximization
                if loss_next >= loss + alpha * learning_rate * np.sum(dlogL_dd2 * dlogL_dd2):
                    break
                learning_rate *= d_factor
                ct += 1

            if ct < max_iters:
                ct_d2 = ct
                smooth_d2 = learning_rate
                loss = loss_next
                self.d2 = d2_plus
            else:
                ct_d2 = np.inf
                smooth_d2 = 0
            loss_d2 = loss

            # set up variables to compute loss in next round
            s2 = np.exp(self.d2)
            s2[0, :] = 1
            s2_norm = (1 / np.sum(s2)) * s2
            beta_s2_penalty = - tau_beta * np.exp(2 * max_gamma) * (s2_norm.T @ diagBetaDeltaBeta).squeeze()
            s2_norm_minus_linv = s2_norm - 1 / L
            s2_penalty = - tau_s * (s2_norm_minus_linv.T @ s2_norm_minus_linv).squeeze()

            dlogL_dchi = 0
            ct_chi = 0
            smooth_chi = 0
            loss_chi = 0

        else:
            # smooth_chi
            ct = 0
            learning_rate = 1
            beta = np.exp(self.gamma)  # variable
            exp_chi = np.exp(self.chi)  # variable
            G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
            GBeta = G @ beta  # variable
            GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])
            lambdainv = 1 / (GBetaBPsi)
            beta_Bpsi = [beta @ b for k, b in enumerate(B_sparse)]
            dlogL_dchi = G * np.vstack(
                [np.sum((lambdainv[k] * beta_Bpsi[k] - 1) * self.Y[k] - (np.eye(L) - G[k]) @ beta_Bpsi[k] * self.dt, axis=1)
                 for k in range(K)])
            while ct < max_iters:
                chi_plus = self.chi + learning_rate * dlogL_dchi

                # set up variables to compute loss
                chi_plus[0, :] = 0
                exp_chi = np.exp(chi_plus)  # variable
                G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
                GBeta = G @ beta  # didnt change
                GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
                lambda_del_t = GBetaBPsi * self.dt
                # compute loss
                log_likelihood = np.sum(np.log(GBetaBPsi) * self.Y - lambda_del_t)
                loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty

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

            dlogL_dgamma = 0
            ct_gamma = 0
            smooth_gamma = 0
            loss_gamma = 0
            dlogL_dd1 = 0
            ct_d1 = 0
            smooth_d1 = 0
            loss_d1 = 0
            dlogL_dd2 = 0
            ct_d2 = 0
            smooth_d2 = 0
            loss_d2 = 0

        ################################################################################################################
        # if time_warping:
        #     # smooth_alpha
        #     ct = 0
        #     learning_rate = 1
        #     y_minus_lambda_del_t = self.Y - lambda_del_t
        #     knots_Bpsinminus1_1 = [
        #         (self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for
        #         time in time_matrix]
        #     knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R * T))]).tocsc() for b_deriv in
        #                            knots_Bpsinminus1_1]
        #     GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
        #     GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
        #     psi_norm_Omega = psi_norm @ self.Omega_psi_B
        #     dlogL_dalpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c * np.vstack(
        #         [0.5 * max(self.time) * self.degree * np.sum(
        #             GBetaBPsiDerivXyLambda * ((self.U_ones[q] - psi_norm) @ hstack([self.V] * R)), axis=1) -
        #          2 * tau_psi * np.sum(psi_norm_Omega * (self.U_ones[q] - psi_norm), axis=1)
        #          for q in range(Q)]).T
        #     # we multiply by max time here because in the likelihood we multiply by max time, so its the derivarive of a constant times a function of alpha.
        #     while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
        #         alpha_plus = self.alpha + learning_rate * dlogL_dalpha
        #
        #         # set up variables to compute loss
        #         exp_alpha_c = (np.exp(alpha_plus) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K,
        #                                                                                    axis=0)
        #         psi = exp_alpha_c @ self.U_ones  # variable
        #         psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        #         time_matrix = 0.5 * max(self.time) * np.hstack(
        #             [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        #         B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]
        #         GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #         lambda_del_t = GBetaBPsi * self.dt
        #         # compute loss
        #         log_likelihood = np.sum(np.log(GBetaBPsi) * self.Y - lambda_del_t)
        #         psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
        #         loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty
        #
        #         # Armijo condition, using Frobenius norm for matrices, but for maximization
        #         if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dalpha, ord='fro') ** 2:
        #             break
        #         learning_rate *= alpha_factor
        #         ct += 1
        #
        #     if ct < max_iters:
        #         ct_alpha = ct
        #         smooth_alpha = learning_rate
        #         loss = loss_next
        #         self.alpha = alpha_plus
        #     else:
        #         ct_alpha = np.inf
        #         smooth_alpha = 0
        #     loss_alpha = loss
        #
        #     # set up variables to compute loss in next round
        #     exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K,
        #                                                                                axis=0)  # now fixed
        #     psi = exp_alpha_c @ self.U_ones  # now fixed
        #     psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # now fixed, called \psi' in the document
        #     time_matrix = 0.5 * max(self.time) * np.hstack(
        #         [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        #     B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in
        #                 time_matrix]  # variable
        #     GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #     lambda_del_t = GBetaBPsi * self.dt
        #     # compute updated penalty
        #     psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
        #
        #     # smooth_zeta
        #     ct = 0
        #     learning_rate = 1
        #     y_minus_lambda_del_t = self.Y - lambda_del_t
        #     knots_Bpsinminus1_1 = [
        #         (self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for
        #         time in time_matrix]
        #     knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R * T))]).tocsc() for b_deriv in
        #                            knots_Bpsinminus1_1]
        #     GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
        #     GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
        #     kappa_norm_Omega = kappa_norm @ self.Omega_psi_B
        #     dlogL_dzeta = kappa_norm[:, 1, np.newaxis] * exp_zeta_c * np.vstack([0.5 * max(self.time) * self.degree *
        #                                                                          np.sum(np.sum(
        #                                                                              GBetaBPsiDerivXyLambda * ((
        #                                                                                                                self.U_ones[
        #                                                                                                                    q] - kappa_norm) @ self.V).flatten(),
        #                                                                              axis=0).reshape((R, T)), axis=1) -
        #                                                                          2 * tau_psi * np.sum(
        #         kappa_norm_Omega * (self.U_ones[q] - kappa_norm), axis=1)
        #                                                                          for q in range(Q)]).T
        #     # we multiply by max time here because in the likelihood we multiply by max time, so its the derivarive of a constant times a function of alpha.
        #     while ct < max_iters:  # otherwise there isn't a good decrement direction/it runs into overflow limitations
        #         zeta_plus = self.zeta + learning_rate * dlogL_dzeta
        #
        #         # set up variables to compute loss
        #         exp_zeta_c = (np.exp(zeta_plus) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R,
        #                                                                                  axis=0)
        #         kappa = exp_zeta_c @ self.U_ones  # variable
        #         kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
        #         time_matrix = 0.5 * max(self.time) * np.hstack(
        #             [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        #         B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]
        #         GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #         lambda_del_t = GBetaBPsi * self.dt
        #         # compute loss
        #         log_likelihood = np.sum(np.log(GBetaBPsi) * self.Y - lambda_del_t)
        #         kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        #         loss_next = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty
        #
        #         # Armijo condition, using Frobenius norm for matrices, but for maximization
        #         if loss_next >= loss + alpha * learning_rate * np.linalg.norm(dlogL_dzeta, ord='fro') ** 2:
        #             break
        #         learning_rate *= alpha_factor
        #         ct += 1
        #
        #     if ct < max_iters:
        #         ct_zeta = ct
        #         smooth_zeta = learning_rate
        #         loss = loss_next
        #         self.zeta = zeta_plus
        #     else:
        #         ct_zeta = np.inf
        #         smooth_zeta = 0
        #     loss_zeta = loss
        #
        #     # set up variables to compute loss in next round
        #     exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R,
        #                                                                              axis=0)  # now fixed
        #     kappa = exp_zeta_c @ self.U_ones  # now fixed
        #     kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # now fixed, called \kappa' in the document
        #     time_matrix = 0.5 * max(self.time) * np.hstack(
        #         [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # now fixed
        #     B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in
        #                 time_matrix]  # now fixed
        #     GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])  # variable
        #     # compute updated penalty
        #     kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        #
        # else:
        dlogL_dalpha = 0
        smooth_alpha = 0
        ct_alpha = 0
        loss_alpha = 0
        dlogL_dzeta = 0
        smooth_zeta = 0
        ct_zeta = 0
        loss_zeta = 0

        result = {
            "dlogL_dgamma": dlogL_dgamma,
            "gamma_loss_increase": loss_gamma - loss_0,
            "smooth_gamma": smooth_gamma,
            "iters_gamma": ct_gamma,
            "dlogL_dd1": dlogL_dd1,
            "d1_loss_increase": loss_d1 - loss_gamma,
            "smooth_d1": smooth_d1,
            "iters_d1": ct_d1,
            "dlogL_dd2": dlogL_dd2,
            "d2_loss_increase": loss_d2 - loss_d1,
            "smooth_d2": smooth_d2,
            "iters_d2": ct_d2,
            "dlogL_dalpha": dlogL_dalpha,
            "alpha_loss_increase": loss_alpha - loss_d2,
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
            "likelihood": loss,
            "beta_s1_penalty": beta_s1_penalty,
            "s1_penalty": s1_penalty,
            "beta_s2_penalty": beta_s2_penalty,
            "s2_penalty": s2_penalty,
            "psi_penalty": psi_penalty,
            "kappa_penalty": kappa_penalty
        }

        return result

    def compute_loss_objects(self, tau_psi, tau_beta, tau_s, time_warping):
        K, L = self.chi.shape
        R, Q = self.zeta.shape
        exp_alpha_c = (np.exp(self.alpha) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, K, axis=0)
        psi = exp_alpha_c @ self.U_ones  # variable
        psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
        exp_zeta_c = (np.exp(self.zeta) @ self.alpha_prime_multiply) + np.repeat(self.alpha_prime_add, R, axis=0)
        kappa = exp_zeta_c @ self.U_ones  # variable
        kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
        time_matrix = 0.5 * max(self.time) * np.hstack( [(psi_norm + kappa_norm[r]) @ self.V for r in range(R)])  # variable
        B_sparse = [BSpline.design_matrix(time, self.knots, self.degree).transpose() for time in time_matrix]  # variable
        maxes = [np.max(self.chi[k][:,np.newaxis] + self.gamma) for k in range(K)]
        sum_exps_chi = [np.sum(np.exp(self.chi[k] - maxes[k])) for k in range(K)]
        sum_exps_chi_plus_gamma_B = [np.sum(np.exp(self.chi[k][:,np.newaxis] + self.gamma - maxes[k]), axis=0)[np.newaxis,:] @ b for k, b in enumerate(B_sparse)]
        log_likelihood = np.sum(np.vstack([(np.log(sum_exps_chi_plus_gamma_B[k]) - np.log(sum_exps_chi[k])) * self.Y[k] -
                                           (1/sum_exps_chi[k] * sum_exps_chi_plus_gamma_B[k]) * self.dt for k in range(K)]))
        if time_warping:
            psi_penalty = - tau_psi * np.sum((psi_norm @ self.Omega_psi_B) * psi_norm)
            kappa_penalty = - tau_psi * np.sum((kappa_norm @ self.Omega_psi_B) * kappa_norm)
        else:
            psi_penalty = 0
            kappa_penalty = 0

        s1 = np.exp(self.d1)
        s1[0, :] = 1
        s1_norm = (1 / np.sum(s1)) * s1
        max_gamma = np.max(self.gamma)
        beta_minus_max = np.exp(self.gamma - max_gamma)
        beta_s1_penalty = - tau_beta * np.exp(2*max_gamma) * (s1_norm.T @ np.sum((beta_minus_max @ self.BDelta1TDelta1BT) * beta_minus_max, axis=1)).squeeze()
        # we can pretend np.exp(2 * max_gamma) is part of the penalty
        s1_norm_minus_linv = s1_norm - 1 / L
        s1_penalty = - tau_s * (s1_norm_minus_linv.T @ s1_norm_minus_linv).squeeze()
        s2 = np.exp(self.d2)
        s2[0, :] = 1
        s2_norm = (1 / np.sum(s2)) * s2
        beta_s2_penalty = - tau_beta * np.exp(2*max_gamma) * (s2_norm.T @ np.sum((beta_minus_max @ self.BDelta2TDelta2BT) * beta_minus_max, axis=1)).squeeze()
        s2_norm_minus_linv = s2_norm - 1 / L
        s2_penalty = - tau_s * (s2_norm_minus_linv.T @ s2_norm_minus_linv).squeeze()

        loss = log_likelihood + psi_penalty + kappa_penalty + beta_s1_penalty + s1_penalty + beta_s2_penalty + s2_penalty

        result = {
            "B_sparse": B_sparse,
            "exp_alpha_c": exp_alpha_c,
            "exp_zeta_c": exp_zeta_c,
            "kappa_norm": kappa_norm,
            "psi_norm": psi_norm,
            "time_matrix": time_matrix,
            "loss": loss,
            "log_likelihood": log_likelihood,
            "psi_penalty": psi_penalty,
            "kappa_penalty": kappa_penalty,
            "beta_s1_penalty": beta_s1_penalty,
            "s1_penalty": s1_penalty,
            "s1_norm": s1_norm,
            "beta_s2_penalty": beta_s2_penalty,
            "s2_penalty": s2_penalty,
            "s2_norm": s2_norm,
            "maxes": maxes,
            "sum_exps_chi": sum_exps_chi,
            "sum_exps_chi_plus_gamma_B": sum_exps_chi_plus_gamma_B,
            "max_gamma": max_gamma,
            "beta_minus_max": beta_minus_max
        }
        return result

    def compute_analytical_grad_time_warping(self, tau_psi, tau_beta, tau_s, time_warping=False):

        # define parameters
        K, L = self.chi.shape
        R, Q = self.zeta.shape
        T = self.time.shape[0]

        # set up variables to compute loss
        objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s, time_warping)
        B_sparse = objects["B_sparse"]
        G = objects["G"]
        GBeta = objects["GBeta"]
        beta = objects["beta"]
        s1_norm = objects["s1_norm"]
        s2_norm = objects["s2_norm"]
        exp_alpha_c = objects["exp_alpha_c"]
        exp_zeta_c = objects["exp_zeta_c"]
        kappa_norm = objects["kappa_norm"]
        psi_norm = objects["psi_norm"]
        time_matrix = objects["time_matrix"]
        lambda_del_t = objects["lambda_del_t"]
        GBetaBPsi = objects["GBetaBPsi"]
        eps = objects["eps"]

        # compute gradients
        lambdainv_y_minus_del_t = 1 / GBetaBPsi * self.Y - self.dt
        exp_KprimeBetaBDeltaT = np.exp(kprime * beta @ self.Delta2BT.T)
        likelihood_component = G.T @ np.vstack([lambdainv_y_minus_del_t[k] @ b.transpose() for k, b in enumerate(B_sparse)])
        s1_component = 2 * s1_norm * beta @ self.BDelta1TDelta1BT
        s2_component = s2_norm * (2 / np.log(1 + eps + exp_KprimeBetaBDeltaT) * exp_KprimeBetaBDeltaT - 1) @ self.Delta2BT
        dlogL_dgamma = beta * (likelihood_component - tau_beta * (s1_component + s2_component))

        dlogL_dd1 = s1_norm * (s1_norm.T - np.eye(L)) @ (
                tau_beta * np.sum((beta @ self.BDelta1TDelta1BT) * beta, axis=1)[:, np.newaxis] +
                2 * tau_s * (s1_norm - 1 / L))

        KprimeBetaBDelta2T = kprime * beta @ self.Delta2BT.T
        dlogL_dd2 = 2 * s2_norm * (s2_norm.T - np.eye(L)) @ (tau_beta / kprime * np.sum(np.log(1 + eps + np.exp(
                    KprimeBetaBDelta2T)) - 1 / 2 * KprimeBetaBDelta2T - np.log(2), axis=1)[:, np.newaxis] + tau_s * (s2_norm - 1 / L))

        if time_warping:
            y_minus_lambda_del_t = self.Y - lambda_del_t
            knots_Bpsinminus1_1 = [
                (self.knots_1 * BSpline.design_matrix(time, self.knots[:-1], (self.degree - 1)).transpose()).tocsc() for
                time in time_matrix]
            knots_Bpsinminus1_2 = [vstack([b_deriv[1:], csr_array((1, R * T))]).tocsc() for b_deriv in
                                   knots_Bpsinminus1_1]
            GBetaBPsiDeriv = np.vstack([GBeta[k] @ (knots_Bpsinminus1_1[k] - knots_Bpsinminus1_2[k]) for k in range(K)])
            GBetaBPsiDerivXyLambda = GBetaBPsiDeriv * y_minus_lambda_del_t
            psi_norm_Omega = psi_norm @ self.Omega_psi_B
            kappa_norm_Omega = kappa_norm @ self.Omega_psi_B
            dlogL_dalpha = psi_norm[:, 1, np.newaxis] * exp_alpha_c * np.vstack(
                [0.5 * max(self.time) * self.degree * np.sum(
                    GBetaBPsiDerivXyLambda * ((self.U_ones[q] - psi_norm) @ hstack([self.V] * R)), axis=1) -
                 2 * tau_psi * np.sum(psi_norm_Omega * (self.U_ones[q] - psi_norm), axis=1)
                 for q in range(Q)]).T

            dlogL_dzeta = kappa_norm[:, 1, np.newaxis] * exp_zeta_c * np.vstack([0.5 * max(self.time) * self.degree *
                                                                                 np.sum(np.sum(
                                                                                     GBetaBPsiDerivXyLambda * ((
                                                                                                                       self.U_ones[
                                                                                                                           q] - kappa_norm) @ self.V).flatten(),
                                                                                     axis=0).reshape((R, T)), axis=1) -
                                                                                 2 * tau_psi * np.sum(
                kappa_norm_Omega * (self.U_ones[q] - kappa_norm), axis=1)
                                                                                 for q in range(Q)]).T
        else:
            dlogL_dalpha = 0
            dlogL_dzeta = 0

        lambdainv = 1 / (GBetaBPsi)
        beta_Bpsi = [beta @ b for k, b in enumerate(B_sparse)]
        dlogL_dchi = G * np.vstack(
            [np.sum((lambdainv[k] * beta_Bpsi[k] - 1) * self.Y[k] - (np.eye(L) - G[k]) @ beta_Bpsi[k] * self.dt, axis=1)
             for k in range(K)])

        return dlogL_dgamma, dlogL_dd1, dlogL_dd2, dlogL_dalpha, dlogL_dzeta, dlogL_dchi

    def compute_grad_chunk(self, name, variable, i, eps, tau_psi, tau_beta, tau_s, loss, time_warping=False):

        J = variable.shape[1]
        grad_chunk = np.zeros(J)

        print(f'Starting {name} gradient chunk at row: {i}')

        for j in range(J):
            orig = variable[i, j]
            variable[i, j] = orig + eps
            objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s, time_warping)
            loss_eps = objects['loss']
            grad_chunk[j] = (loss_eps - loss) / eps
            variable[i, j] = orig

        print(f'Completed {name} gradient chunk at row: {i}')

        return grad_chunk

    def compute_numerical_grad_time_warping_parallel(self, tau_psi, tau_beta, tau_s, time_warping=False):

        eps = 1e-4
        # define parameters
        K, L = self.chi.shape
        R = self.trials

        objects = self.compute_loss_objects(tau_psi, tau_beta, tau_s)
        loss = objects['loss']
        pool = mp.Pool()

        # alpha gradient
        alpha_async_results = [
            pool.apply_async(self.compute_grad_chunk, args=('alpha', self.alpha, k, eps, tau_psi, tau_beta, loss)) for k
            in range(K)]

        # d1 gradient
        d1_async_results = [
            pool.apply_async(self.compute_grad_chunk, args=('d1', self.d1, l, eps, tau_psi, tau_beta, loss)) for
            l in range(L)]

        # d2 gradient
        d2_async_results = [
            pool.apply_async(self.compute_grad_chunk, args=('d2', self.d2, l, eps, tau_psi, tau_beta, loss)) for
            l in range(L)]

        if time_warping:
            # zeta gradient
            zeta_async_results = [
                pool.apply_async(self.compute_grad_chunk, args=('zeta', self.zeta, r, eps, tau_psi, tau_beta, loss)) for
                r in range(R)]

            # gamma gradient
            gamma_async_results = [
                pool.apply_async(self.compute_grad_chunk, args=('gamma', self.gamma, l, eps, tau_psi, tau_beta, loss))
                for l in range(L)]

        # chi gradient
        chi_async_results = [
            pool.apply_async(self.compute_grad_chunk, args=('chi', self.chi, k, eps, tau_psi, tau_beta, loss)) for k in
            range(K)]

        pool.close()
        pool.join()

        alpha_grad = np.vstack([r.get() for r in alpha_async_results])
        d1_grad = np.vstack([r.get() for r in d1_async_results])
        d2_grad = np.vstack([r.get() for r in d2_async_results])
        if time_warping:
            zeta_grad = np.vstack([r.get() for r in zeta_async_results])
            gamma_grad = np.vstack([r.get() for r in gamma_async_results])
        else:
            zeta_grad = 0
            gamma_grad = 0
        chi_grad = np.vstack([r.get() for r in chi_async_results])

        return gamma_grad, d1_grad, d2_grad, alpha_grad, zeta_grad, chi_grad
