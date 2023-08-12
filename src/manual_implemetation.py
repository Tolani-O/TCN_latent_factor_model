import numpy as np

def log_prob(Y, B, G, beta, dt):
    # Forward pass as before
    GBetaB = np.dot(np.dot(G, beta), B)
    lambda_del_t = np.exp(GBetaB) * dt
    log_likelihood = np.sum(GBetaB * Y - lambda_del_t)

    # Manual gradients
    y_minus_lambdadt_times_B = np.dot((Y - lambda_del_t), B.T)
    dlogL_dG = np.dot(y_minus_lambdadt_times_B, beta.T)
    dlogL_dbeta = np.dot(G.T, y_minus_lambdadt_times_B)

    return -log_likelihood, -dlogL_dG, -dlogL_dbeta

def compute_lambda(B, G, beta):
    GBetaB = np.dot(np.dot(G, beta), B)
    lambda_ = np.exp(GBetaB)
    return lambda_

def compute_latent_factors(B, beta):
    N = np.dot(beta, B)
    return N

def compute_numerical_grad(Y, B, G, beta, dt, eps=1e-6):
    """Compute numerical gradient of func w.r.t G"""
    G_grad = np.zeros_like(G)
    beta_grad = np.zeros_like(beta)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            lk1, _, _ = log_prob(Y, B, G, beta, dt)
            orig = G[i, j]
            G[i,j] = orig + eps
            lk2, _, _ = log_prob(Y, B, G, beta, dt)
            G[i, j] = orig
            G_grad[i,j] = (lk1 - lk2) / eps

    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            lk1, _, _ = log_prob(Y, B, G, beta, dt)
            orig = beta[i, j]
            beta[i,j] = orig + eps
            lk2, _, _ = log_prob(Y, B, G, beta, dt)
            beta[i, j] = orig
            beta_grad[i,j] = (lk1 - lk2) / eps

    return -G_grad, -beta_grad
