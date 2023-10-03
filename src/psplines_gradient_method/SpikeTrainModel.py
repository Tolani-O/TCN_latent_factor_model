import numpy as np
from src.psplines_gradient_method.general_functions import create_first_diff_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bspline_matrix


class SpikeTrainModel:
    def __init__(self, Y, time):
        # variables
        self.G = None
        self.beta = None
        self.d = None

        # parameters
        self.Y = Y
        self.time = time
        self.J = None
        self.V = None
        self.Omega_beta = None
        self.degree = None

    def initialize(self, L, degree):

        # parameters
        self.degree = degree

        # latent factor b-spline matrix. Re-using the self.V matrix. Coefficients would be from beta
        self.V = generate_bspline_matrix(self.time, self.degree)
        K = self.Y.shape[0]
        P = self.V.shape[0]
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
        P = self.V.shape[0]

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

