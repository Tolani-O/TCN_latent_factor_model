import numpy as np
from src.allen_data import EcephysAnalyzer
from src.psplines_gradient_method.manual_implemetation import log_prob, log_obj, log_obj_with_backtracking_line_search
from src.psplines_gradient_method.general_functions import compute_lambda, \
    create_first_diff_matrix, create_second_diff_matrix, plot_binned, plot_spikes
from src.psplines_gradient_method.generate_bsplines import generate_bsplines
import matplotlib.pyplot as plt


degree, L = 3, 2
responsive_units = [85, 143, 76, 107, 116, 95, 46, 222, 73, 118, 15, 43, 34, 13, 17, 18, 79, 172, 221,
                    23, 29, 31, 54, 211, 47, 148, 90, 229, 28, 50, 8, 99, 48, 101, 39, 91, 127, 137,
                    11, 66, 117, 55, 20, 87, 27, 22, 170, 21, 2, 30, 19, 12, 191]
non_responsive_units = [238, 242, 64, 185, 138, 98, 10, 182, 52, 105, 202, 175, 132, 204, 210, 219, 240,
                        239, 205, 102, 75, 140, 203, 111, 217, 233, 196, 230, 187, 235, 192, 206, 60, 178,
                        228, 128, 173, 84, 156, 1, 180, 168, 214, 174, 141, 186, 215, 164, 106, 234, 130,
                        197, 241, 243, 149, 200, 220, 181, 227, 126]
presentation_ids = 49417
# Manual Implementation
self = EcephysAnalyzer().initialize()
binned, spikes, time = self.sample_data(unit_ids=responsive_units + non_responsive_units, presentation_ids=presentation_ids)
K = binned.shape[0]
dt = round(time[1] - time[0], 3)
# make plots
spikes['unit_id'] = spikes['unit_id'].astype(str)
spikes.plot(x='time_since_stimulus_presentation_onset', y='unit_id', kind='scatter', s=1, yticks=[])
plt.show()
binned_spikes = np.where(binned >= 1)
plot_spikes(binned_spikes)


Y = binned
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

# Training parameters
num_epochs = 2500
Omega = create_first_diff_matrix(P)

# Training hyperparameters
tau_beta = 80
tau_G = 1

# # Training hyperparameters
# num_epochs = 4000
# beta_tausq = 80*np.ones(L) # 10*np.square(np.random.rand(L))
# G_eta = 10
# smooth = 2000
# G_smooth = 400
# Omega = create_second_diff_matrix(P)

G_grads_norm = []
beta_grads_norm = []
d_grads_norm = []
G_smooths = []
beta_smooths = []
d_smooths = []
losses = []

d_loss_increase = []
d_next_likelihood = []
G_loss_increase = []
G_next_likelihood = []
beta_loss_increase = []
beta_next_likelihood = []
smooth_d = []
smooth_G = []
smooth_beta = []
iters_d = []
iters_G = []
iters_beta = []
for epoch in range(num_epochs):
    # Forward pass and gradient computation
    result = log_obj_with_backtracking_line_search(Y, B, d, G, beta, Omega, tau_beta, tau_G, dt)
    loss = result["loss"]
    dd = result["dLogL_dd"]
    dG = result["dlogL_dG"]
    dbeta = result["dlogL_dbeta"]
    log_likelihood = result["log_likelihood"]
    beta_penalty = result["beta_penalty"]
    G_penalty = result["G_penalty"]

    G_grads_norm.append(np.linalg.norm(dG, ord=2))
    beta_grads_norm.append(np.linalg.norm(dbeta, ord=2))
    d_grads_norm.append(np.linalg.norm(dd, ord=2))
    d_loss_increase.append(result["d_loss_increase"])
    d_next_likelihood.append(result["d_likelihood_next"])
    G_loss_increase.append(result["G_loss_increase"])
    G_next_likelihood.append(result["G_likelihood_next"])
    beta_loss_increase.append(result["beta_loss_increase"])
    beta_next_likelihood.append(result["beta_likelihood_next"])
    smooth_d.append(result["smooth_d"])
    smooth_G.append(result["smooth_G"])
    smooth_beta.append(result["smooth_beta"])
    iters_d.append(result["iters_d"])
    iters_G.append(result["iters_G"])
    iters_beta.append(result["iters_beta"])

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

    # Update parameters using gradients
    d = result["d_plus"]
    G = result["G_plus"]
    beta = result["beta_plus"]
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

lambda_manual = compute_lambda(B, d, G, beta)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
for i in range(K):
    plt.plot(time, lambda_manual[i, :] + i*2)
plt.show()

latent_factors_manual = beta @ B
for i in range(L):
    plt.plot(np.concatenate([[time[0] - dt], time]), beta[i, :])
    plt.plot(time, latent_factors_manual[i, :])
    plt.title(f'Factor [{i}, :]')
plt.show()

# for i in range(L):
#     plt.plot(time, latent_factors_manual[i, :])
# plt.show()
