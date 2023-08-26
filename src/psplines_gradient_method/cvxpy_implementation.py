import cvxpy as cp
import numpy as np

from src.psplines_gradient_method.general_functions import create_first_diff_matrix


def minimize_log_obj(Y, B, K, L, P, beta_tausq, dt):
    G = cp.Variable((K, P))
    beta = cp.Variable((L, P))
    d = cp.Variable(K)

    J = np.ones_like(Y)
    Omega = create_first_diff_matrix(P)
    diagdJ_plus_GBetaB = cp.diag(d) @ J + G @ beta @ B

    log_likelihood = cp.sum(cp.multiply(diagdJ_plus_GBetaB, Y) - cp.exp(diagdJ_plus_GBetaB) * dt)
    beta_penalty = - cp.norm(Omega @ beta.T @ cp.diag(cp.sqrt(beta_tausq)), 'fro')
    # G_penalty = - cp.pnorm(G, 1)
    loss = log_likelihood + beta_penalty  # + G_penalty

    objective = cp.Minimize(-loss)
    problem = cp.Problem(objective)

    # Solving the problem
    problem.solve()

    return beta.value, G.value, d.value, loss.value
