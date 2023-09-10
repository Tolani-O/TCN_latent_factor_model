import numpy as np
import src.simulate_data as sd
from src.psplines_gradient_method.manual_implemetation import log_prob, log_obj, log_obj_with_backtracking_line_search, \
    log_obj_with_backtracking_line_search_and_time_warping
from src.psplines_gradient_method.general_functions import compute_lambda, compute_numerical_grad, \
    create_first_diff_matrix, create_second_diff_matrix, plot_binned, plot_spikes, plot_intensity_and_latents, \
    create_masking_matrix
from src.psplines_gradient_method.generate_bsplines import generate_bspline_functions, generate_bspline_matrix, \
    bspline_deriv_multipliers
import matplotlib.pyplot as plt


K, degree, T = 100, 3, 200
intensity_type = ('constant', '1peak', '2peaks')
L = len(intensity_type) - 1
# base firing rate
time = np.arange(0, T, 1) / 100
dt = time[1] - time[0]

latent_factors = sd.generate_latent_factors(time, intensity_type=intensity_type)
np.random.seed(0)
intensity, binned, spikes = sd.generate_spike_trains(latent_factors, (0.1, 0.13, 0.13), (-3, -3, -3),
                                                     (1 / 3, 1 / 3, 1 / 3), K)
K = binned.shape[0]

# plot_intensity_and_latents(time, latent_factors, intensity)
# plot_binned(binned)
# plot_spikes(spikes)

# Manual Implementation
Y = binned  # K x T
J = np.ones_like(Y)
B_func_n = generate_bspline_functions(time, degree)  # time warping b-spline functions
P = len(B_func_n)
time_matrix = time[np.newaxis, :]
V = generate_bspline_matrix(B_func_n, time_matrix) # time warping b-spline matrix. Coefficients would be from psi
B_func_nminus1 = generate_bspline_functions(time, degree, True)  # for psi derivatives
knots_1, knots_2 = bspline_deriv_multipliers(time, degree)  # for psi derivatives
Q = V.shape[0]
P = len(time) + degree - 1
# start = 190
# num_basis = 10
# for i in range(num_basis):
#     plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
# plt.show()

# G
np.random.seed(0)
G_star = np.random.rand(K, K*L)
mask_G = create_masking_matrix(K, L)
G_star = G_star * mask_G

# beta
np.random.seed(0)
beta = np.random.rand(L, P)
mask_beta = create_masking_matrix(K, P)
I_beta_P = np.vstack([np.eye(P)] * K)
I_beta_L = np.vstack([np.eye(L)] * K)

# d
np.random.seed(0)
d = np.random.rand(K)

# psi
np.random.seed(0)
psi = np.random.rand(K, Q)
mask_psi = np.hstack([np.eye(K)] * Q)
J_psi = create_masking_matrix(Q, K).T

# Training parameters
num_epochs = 2500
Omega_beta = create_first_diff_matrix(P)
Omega_psi = create_first_diff_matrix(Q)


# Training hyperparameters
tau_psi = 30
tau_beta = 80
tau_G = 2

# # Training hyperparameters
# num_epochs = 4000
# beta_tausq = 80*np.ones(L) # 10*np.square(np.random.rand(L))
# G_eta = 10
# smooth = 2000
# G_smooth = 400
# Omega = create_second_diff_matrix(P)

psi_grads_norm = []
G_grads_norm = []
beta_grads_norm = []
d_grads_norm = []
G_smooths = []
beta_smooths = []
d_smooths = []
losses = []

psi_loss_increase = []
d_loss_increase = []
G_loss_increase = []
beta_loss_increase = []
smooth_d = []
smooth_G = []
smooth_beta = []
iters_d = []
iters_G = []
iters_beta = []
for epoch in range(num_epochs):
    # Forward pass and gradient computation

    # result = log_obj(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt)
    result = log_obj_with_backtracking_line_search_and_time_warping(
        Y, J, B_func_n, B_func_nminus1, knots_1, knots_2, V,
        d, G_star, mask_G, beta, mask_beta, I_beta_P, I_beta_L, psi, mask_psi, J_psi, Omega_beta, Omega_psi,
        tau_psi, tau_beta, tau_G)
    loss = result["loss"]
    dd = result["dLogL_dd"]
    dG_star = result["dlogL_dG"]
    dbeta = result["dlogL_dbeta"]
    dpsi = result["dlogL_dpsi"]
    log_likelihood = result["log_likelihood"]
    beta_penalty = result["beta_penalty"]
    G_penalty = result["G_penalty"]
    psi_penalty = result["psi_penalty"]


    psi_grads_norm.append(np.linalg.norm(dpsi, ord=2))
    G_grads_norm.append(np.linalg.norm(dG_star, ord=2))
    beta_grads_norm.append(np.linalg.norm(dbeta, ord=2))
    d_grads_norm.append(np.linalg.norm(dd, ord=2))
    d_loss_increase.append(result["d_loss_increase"])
    G_loss_increase.append(result["G_loss_increase"])
    beta_loss_increase.append(result["beta_loss_increase"])
    psi_loss_increase.append(result["psi_loss_increase"])

    smooth_d.append(result["smooth_d"])
    smooth_G.append(result["smooth_G"])
    smooth_beta.append(result["smooth_beta"])
    iters_d.append(result["iters_d"])
    iters_G.append(result["iters_G"])
    iters_beta.append(result["iters_beta"])

    if epoch > 0:
        G_smooths.append(np.linalg.norm(dG_star - prev_dG_star, ord=2) / np.linalg.norm(G_star - prev_G_star, ord=2))
        beta_smooths.append(np.linalg.norm(dbeta - prev_dbeta, ord=2) / np.linalg.norm(beta - prev_beta, ord=2))
        d_smooths.append(np.linalg.norm(dd - prev_dd, ord=2) / np.linalg.norm(d - prev_d, ord=2))

    prev_G_star = np.copy(G_star)
    prev_dG_star = np.copy(dG_star)
    prev_beta = np.copy(beta)
    prev_dbeta = np.copy(dbeta)
    prev_d = np.copy(d)
    prev_dd = np.copy(dd)
    prev_psi = np.copy(psi)
    prev_dpsi = np.copy(dpsi)

    # Update parameters using gradients
    d = result["d_plus"]
    G_star = result["G_star_plus"]
    beta = result["beta_plus"]
    psi = result["psi_plus"]
    # Store losses and gradients
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

num_epochs = len(losses)
losses = np.array(losses)
G_smooths = np.array(G_smooths)
beta_smooths = np.array(beta_smooths)
d_smooths = np.array(d_smooths)
smooth_G = np.array(smooth_G)
smooth_beta = np.array(smooth_beta)
smooth_d = np.array(smooth_d)
iters_d = np.array(iters_d)
iters_G = np.array(iters_G)
iters_beta = np.array(iters_beta)
plt.plot(np.arange(0, num_epochs), losses[0:])
plt.title('Losses')
plt.show()
plt.plot(np.arange(1, num_epochs), G_smooths)
plt.title('G Smooths Numeric')
plt.show()
plt.plot(np.arange(1, num_epochs), beta_smooths)
plt.title('Beta Smooths Numeric')
plt.show()
plt.plot(np.arange(1, num_epochs), d_smooths)
plt.title('d Smooths Numeric')
plt.show()
plt.plot(np.arange(0, num_epochs), 1 / smooth_G)
plt.title('G Smooths Line Search')
plt.show()
plt.plot(np.arange(0, num_epochs), 1 / smooth_beta)
plt.title('Beta Smooths  Line Search')
plt.show()
plt.plot(np.arange(0, num_epochs), 1 / smooth_d)
plt.title('d Smooths  Line Search')
plt.show()
plt.plot(np.arange(0, num_epochs), iters_G)
plt.title('G Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), iters_beta)
plt.title('Beta Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), iters_d)
plt.title('d Iters')
plt.show()

combined = np.concatenate([losses[:, np.newaxis], smooth_G[:, np.newaxis], smooth_beta[:, np.newaxis], smooth_d[:, np.newaxis],
                            iters_G[:, np.newaxis], iters_beta[:, np.newaxis], iters_d[:, np.newaxis]], axis=1)
d_loss_increase = np.array(d_loss_increase)[:, np.newaxis]
G_loss_increase = np.array(G_loss_increase)[:, np.newaxis]

lambda_manual = compute_lambda(B, d, G, beta)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
np.mean(np.square(intensity - lambda_manual))
for i in range(K):
    plt.plot(time, lambda_manual[i, :] + i * 10)
plt.show()

latent_factors_manual = beta @ B
for i in range(L):
    plt.plot(np.concatenate([[time[0] - dt], time]), beta[i, :])
    plt.plot(time, latent_factors_manual[i, :])
    plt.title(f'Factor [{i}, :]')
plt.show()

G_and_d = np.concatenate([G, d[:, np.newaxis]], axis=1)
