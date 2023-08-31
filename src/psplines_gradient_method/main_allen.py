import numpy as np
from src.allen_data import EcephysAnalyzer
from src.psplines_gradient_method.manual_implemetation import log_prob, log_obj
from src.psplines_gradient_method.general_functions import compute_lambda, compute_latent_factors, \
    compute_numerical_grad, create_first_diff_matrix, create_second_diff_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bsplines
import matplotlib.pyplot as plt


degree, L = 3, 3
# Manual Implementation
analyzer = EcephysAnalyzer()
binned = analyzer.run()
K = binned.shape[1]
time = binned.index.values
dt = round(time[1] - time[0], 3)

Y = binned.to_numpy().T
B = generate_bsplines(time, degree)  # T x T. The coefficient (beta) will be regularized
P = B.shape[0]
# start = 190
# num_basis = 10
# for i in range(num_basis):
#     plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
# plt.show()

np.random.seed(0)
G = np.random.rand(K, L)
np.random.seed(0)
beta = np.random.rand(L, P)
np.random.seed(0)
d = np.random.rand(K)

# Training hyperparameters
num_epochs = 40000
beta_tausq = 10*np.ones(L) # 10*np.square(np.random.rand(L))
G_eta = 2
smooth = 1000
G_smooth = 200
Omega = create_first_diff_matrix(P)

# # Training hyperparameters
# num_epochs = 4000
# beta_tausq = 80*np.ones(L) # 10*np.square(np.random.rand(L))
# G_eta = 10
# smooth = 2000
# G_smooth = 400
# Omega = create_second_diff_matrix(P)

G_grads = []
beta_grads = []
d_grads = []
tausq_grads = []
G_smooths = []
beta_smooths = []
d_smooths = []
losses = []
eps = 1e-4
for epoch in range(num_epochs):
    # Forward pass and gradient computation

    # tausq = 2*tausq
    # loss, dd, dG, dbeta, dtausq = log_prob(Y, B, d, G, beta, tausq, dt)
    #
    # # # verify gradient using finite difference
    # # dd_num, dG_num, dbeta_num, dtausq_num = compute_numerical_grad(Y, B, d, G, beta, tausq, dt, log_prob, eps)
    # # dd_error = np.mean(np.square(dd - dd_num))
    # # dG_error = np.mean(np.square(dG - dG_num))
    # # dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    # # dtausq_error = np.mean(np.square(dtausq - dtausq_num))
    # # d_grads.append(dd_error)
    # # G_grads.append(dG_error)
    # # beta_grads.append(dbeta_error)
    # # tausq_grads.append(dtausq_error)
    # # print(f"Epoch {epoch}, dd_error {dd_error}, dG_error {dG_error}, dbeta_error {dbeta_error}, dtausq_error {dtausq_error}")

    result = log_obj(Y, B, d, G, beta, Omega, beta_tausq, G_eta, G_smooth, smooth, dt)
    loss = result["loss"]
    dd = result["dLogL_dd"]
    dG = result["dlogL_dG"]
    dbeta = result["dlogL_dbeta"]
    log_likelihood = result["log_likelihood"]
    beta_penalty = result["beta_penalty"]
    G_penalty = result["G_penalty"]

    if epoch > 0:
        G_smooths.append(np.linalg.norm(dG - prev_dG, ord=2) / np.linalg.norm(G - prev_G, ord=2))
        beta_smooths.append(np.linalg.norm(dbeta - prev_dbeta, ord=2) / np.linalg.norm(beta - prev_beta, ord=2))
        d_smooths.append(np.linalg.norm(dd - prev_dd, ord=2) / np.linalg.norm(d - prev_d, ord=2))

    prev_G = np.copy(G)
    prev_dG = np.copy(dG)
    prev_beta = np.copy(beta)
    prev_dbeta = np.copy(dbeta)
    prev_d = np.copy(d)
    prev_dd = np.copy(dd)

    # # verify gradient using finite difference
    # dd_num, dG_num, dbeta_num, _ = compute_numerical_grad(Y, B, d, G, beta, tausq, dt, log_obj, eps)
    # dd_error = np.mean(np.square(dd - dd_num))
    # dG_error = np.mean(np.square(dG - dG_num))
    # dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    # d_grads.append(dd_error)
    # G_grads.append(dG_error)
    # beta_grads.append(dbeta_error)
    # print(f"Epoch {epoch}, dd_error {dd_error}, dG_error {dG_error}, dbeta_error {dbeta_error}")

    # Update parameters using gradients
    d = result["d_plus"]
    G = result["G_plus"]
    beta = result["beta_plus"]
    # d = np.copy(dd)
    # G = np.copy(G_plus)
    # beta = np.copy(beta_plus)
    # tausq -= learning_rate * dtausq
    # Store losses and gradients
    losses.append(loss)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

plt.plot(np.arange(0, num_epochs), losses[0:])
plt.show()
plt.plot(np.arange(1, num_epochs), G_smooths)
plt.show()
plt.plot(np.arange(1, num_epochs), beta_smooths)
plt.show()
plt.plot(np.arange(1, num_epochs), d_smooths)
plt.show()

lambda_manual = compute_lambda(B, d, G, beta)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
for i in range(K):
    plt.plot(time, lambda_manual[i, :] + i*10)
plt.show()

latent_factors_manual = compute_latent_factors(B, beta)
for i in range(L):
    plt.plot(np.concatenate([[time[0]-dt], time]), beta[i, :])
    plt.plot(time, latent_factors_manual[i, :])
    plt.title(f'beta[{i}, :]')
    plt.show()

# for i in range(L):
#     plt.plot(time, latent_factors_manual[i, :])
# plt.show()
