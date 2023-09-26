import numpy as np
from src.psplines_gradient_method.generate_bsplines import generate_bspline_matrix
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel
from src.psplines_gradient_method.general_functions import compute_lambda, plot_binned, plot_spikes
import matplotlib.pyplot as plt

self = DataAnalyzer().initialize()
binned, time = self.sample_data()
binned_spikes = np.where(binned >= 1)
# plot_spikes(binned_spikes)
intensity, latent_factors = self.intensity, self.latent_factors
# plot_binned(binned)
# self.plot_intensity_and_latents()

Y = binned  # K x T
degree = 3
L = self.latent_factors.shape[0] - 1

# Training parameters
num_epochs = 2500

# Training hyperparameters
tau_psi = 80
tau_beta = 80
tau_G = 2

model = SpikeTrainModel(Y, time).initialize_for_time_warping(L, degree)

losses = []

psi_loss_increase = []
d_loss_increase = []
G_loss_increase = []
beta_loss_increase = []

psi_learning_rate = []
beta_learning_rate = []
G_learning_rate = []
d_learning_rate = []

psi_iters = []
beta_iters = []
G_iters = []
d_iters = []

for epoch in range(num_epochs):
    # Forward pass and gradient computation

    result = model.log_obj_with_backtracking_line_search_and_time_warping(tau_psi, tau_beta, tau_G)
    loss = result["loss"]
    psi_penalty = result["psi_penalty"]
    beta_penalty = result["beta_penalty"]
    G_penalty = result["G_penalty"]
    dpsi = result["dlogL_dpsi"]
    dbeta = result["dlogL_dbeta"]
    dG_star = result["dlogL_dG"]
    dd = result["dlogL_dd"]

    losses.append(loss)

    psi_loss_increase.append(result["psi_loss_increase"])
    beta_loss_increase.append(result["beta_loss_increase"])
    G_loss_increase.append(result["G_loss_increase"])
    d_loss_increase.append(result["d_loss_increase"])

    psi_learning_rate.append(result["smooth_psi"])
    beta_learning_rate.append(result["smooth_beta"])
    G_learning_rate.append(result["smooth_G"])
    d_learning_rate.append(result["smooth_d"])

    psi_iters.append(result["iters_psi"])
    beta_iters.append(result["iters_beta"])
    G_iters.append(result["iters_G"])
    d_iters.append(result["iters_d"])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

# num_epochs = len(losses)
# losses = np.array(losses)
# G_smooths = np.array(G_smooths)
# beta_smooths = np.array(beta_smooths)
# d_smooths = np.array(d_smooths)
# smooth_G = np.array(smooth_G)
# smooth_beta = np.array(smooth_beta)
# smooth_d = np.array(smooth_d)
# iters_d = np.array(iters_d)
# iters_G = np.array(iters_G)
# iters_beta = np.array(iters_beta)
# plt.plot(np.arange(0, num_epochs), losses[0:])
# plt.title('Losses')
# plt.show()
# plt.plot(np.arange(1, num_epochs), G_smooths)
# plt.title('G Smooths Numeric')
# plt.show()
# plt.plot(np.arange(1, num_epochs), beta_smooths)
# plt.title('Beta Smooths Numeric')
# plt.show()
# plt.plot(np.arange(1, num_epochs), d_smooths)
# plt.title('d Smooths Numeric')
# plt.show()
# plt.plot(np.arange(0, num_epochs), 1 / smooth_G)
# plt.title('G Smooths Line Search')
# plt.show()
# plt.plot(np.arange(0, num_epochs), 1 / smooth_beta)
# plt.title('Beta Smooths  Line Search')
# plt.show()
# plt.plot(np.arange(0, num_epochs), 1 / smooth_d)
# plt.title('d Smooths  Line Search')
# plt.show()
# plt.plot(np.arange(0, num_epochs), iters_G)
# plt.title('G Iters')
# plt.show()
# plt.plot(np.arange(0, num_epochs), iters_beta)
# plt.title('Beta Iters')
# plt.show()
# plt.plot(np.arange(0, num_epochs), iters_d)
# plt.title('d Iters')
# plt.show()

time_matrix = model.psi @ model.V
B_psi = generate_bspline_matrix(model.B_func_n, time_matrix)
lambda_manual = compute_lambda(B_psi, model.d, model.G_star, model.beta)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
np.mean(np.square(intensity - lambda_manual))
for i in range(model.Y.shape[0]):
    plt.plot(time, lambda_manual[9, :] + i * 2)
plt.show()

latent_factors_manual = model.beta @ model.V
for i in range(L):
    # plt.plot(np.concatenate([[time[0] - dt], time]), beta[i, :])
    plt.plot(time, latent_factors_manual[i, :])
    plt.title(f'Factor [{i}, :]')
plt.show()

time_warping_matrix = model.psi @ model.V
for i in range(model.Y.shape[0]):
    plt.plot(time, time_warping_matrix[8, :]+ i * 0.01)
plt.show()

G_and_d = np.concatenate([G, d[:, np.newaxis]], axis=1)
