import numpy as np
import src.simulate_data as sd
from src.psplines_gradient_method.manual_implemetation import log_prob, compute_lambda, compute_latent_factors, \
    compute_numerical_grad, log_obj, create_precision_matrix, create_first_diff_matrix, create_second_diff_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bsplines
import matplotlib.pyplot as plt
from numpy.linalg import det

K, degree, T = 100, 3, 200
intensity_type = ('constant', '1peak', '2peaks')
L = len(intensity_type)
# Training hyperparameters
learning_rate = 1e-3
num_epochs = 50000
# base firing rate
time = np.arange(0, T, 1)/100
dt = time[1] - time[0]

latent_factors = sd.generate_latent_factors(time, intensity_type=intensity_type)
np.random.seed(0)
intensity, binned, spikes = sd.generate_spike_trains(latent_factors, (0.1, 0.13, 0.13), (-3, -3, -3), (1/3, 1/3, 1/3), K)
K = binned.shape[0]

# sd.plot_intensity_and_spikes(time, latent_factors, intensity, binned, spikes)

# Manual Implementation
Y = binned  # K x T
B = generate_bsplines(time, degree)  # T x T. The coefficient (beta) will be regularized
# start = 190
# num_basis = 10
# for i in range(num_basis):
#     plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
# plt.show()

np.random.seed(0)
G = np.random.rand(K, L)
np.random.seed(0)
beta = np.random.rand(L, B.shape[0])
np.random.seed(0)
d = np.random.rand(K)
tausq = 200*np.ones(L) # 10*np.square(np.random.rand(L))

G2 = np.copy(G)
beta2 = np.copy(beta)
d2 = np.copy(d)
tausq2 = 50*np.ones(L)

G_grads = []
beta_grads = []
d_grads = []
losses = []
G_grads2 = []
beta_grads2 = []
d_grads2 = []
losses2 = []
eps=1e-4
for epoch in range(num_epochs):

    L, P = beta.shape
    J = np.ones_like(Y)


    # method 1
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G @ beta @ B
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    log_likelihood = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T
    dLogL_dd = np.sum(y_minus_lambdadt, axis=1)
    dlogL_dG = y_minus_lambdadt_times_B @ beta.T
    dd, dG = -dLogL_dd, -dlogL_dG

    Omega = create_precision_matrix(P)
    BetaOmegaBeta = beta @ Omega @ beta.T
    log_prior_beta = 0.5 * (L * np.log((2 * np.pi) ** (-P) * det(Omega)) +
                            P * np.sum(np.log(tausq)) -
                            tausq.T @ np.diag(BetaOmegaBeta))

    dlogL_dbeta = (G.T @ y_minus_lambdadt_times_B -
                   tausq[:, np.newaxis] * beta @ Omega)
    dbeta = -dlogL_dbeta


    # method 2
    diagdJ_plus_GBetaB = d2[:, np.newaxis] * J + G2 @ beta2 @ B
    lambda_del_t = np.exp(diagdJ_plus_GBetaB) * dt
    log_likelihood2 = np.sum(diagdJ_plus_GBetaB * Y - lambda_del_t)
    y_minus_lambdadt = Y - lambda_del_t
    y_minus_lambdadt_times_B = y_minus_lambdadt @ B.T
    dLogL_dd = np.sum(y_minus_lambdadt, axis=1)
    dlogL_dG = y_minus_lambdadt_times_B @ beta2.T
    dd2, dG2 = -dLogL_dd, -dlogL_dG

    # Omega2 = create_first_diff_matrix(P)
    Omega2 = create_second_diff_matrix(P)
    Omega2 = Omega2.T @ Omega2
    BetaOmegaBeta2 = beta2 @ Omega2 @ beta2.T
    log_prior_beta2 = - tausq2.T @ np.diag(BetaOmegaBeta2)

    dlogL_dbeta = (G2.T @ y_minus_lambdadt_times_B -
                   2 * tausq2[:, np.newaxis] * beta2 @ Omega2)
    dbeta2 = -dlogL_dbeta


    # Update parameters using gradients
    d -= learning_rate * dd
    G -= learning_rate * dG
    beta -= learning_rate * dbeta
    d2 -= learning_rate * dd2
    G2 -= learning_rate * dG2
    beta2 -= learning_rate * dbeta2
    losses.append(log_likelihood + log_prior_beta)
    losses2.append(log_likelihood2 + log_prior_beta2)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, method1: likelihood {log_likelihood}, penalty {log_prior_beta}, loss {log_likelihood + log_prior_beta}")
        print(f"Epoch {epoch}, method2: likelihood {log_likelihood2}, penalty {log_prior_beta2}, loss {log_likelihood2 + log_prior_beta2}")
        print(f"Epoch {epoch}, Divergences: beta {np.sum(beta - beta2)}, d {np.sum(d - d2)}, G {np.sum(G - G2)}")


plt.plot(np.arange(0, num_epochs), losses)
plt.show()

plt.plot(np.arange(0, num_epochs), losses2)
plt.show()

lambda_manual = compute_lambda(B, d, G, beta)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
np.mean(np.square(intensity - lambda_manual))
for i in range(K):
    plt.plot(time, lambda_manual[i, :] + i*10)
plt.show()

lambda_manual2 = compute_lambda(B, d2, G2, beta2)
avg_lambda_manual = np.mean(lambda_manual2, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
np.mean(np.square(intensity - lambda_manual2))
for i in range(K):
    plt.plot(time, lambda_manual2[i, :] + i*10)
plt.show()

latent_factors_manual = compute_latent_factors(B, beta)
for i in range(L):
    plt.plot(np.concatenate([[time[0]-dt], time]), beta[i, :])
    plt.plot(time, latent_factors_manual[i, :])
    plt.title(f'beta[{i}, :]')
    plt.show()

latent_factors_manual2 = compute_latent_factors(B, beta2)
for i in range(L):
    plt.plot(np.concatenate([[time[0]-dt], time]), beta2[i, :])
    plt.plot(time, latent_factors_manual2[i, :])
    plt.title(f'beta[{i}, :]')
    plt.show()
