import numpy as np
from scipy.interpolate import BSpline
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel
from src.psplines_gradient_method.general_functions import plot_binned, plot_spikes
import matplotlib.pyplot as plt
import time

self = DataAnalyzer().initialize(K=600, max_offset=0)
binned, stim_time = self.sample_data()
binned_spikes = np.where(binned >= 1)
# plot_spikes(binned_spikes)
intensity, latent_factors = self.intensity, self.latent_factors
# plot_binned(binned)
# self.plot_intensity_and_latents()

Y = binned  # K x T
degree = 3
L = 3#self.latent_factors.shape[0] - 1
model = SpikeTrainModel(Y, stim_time).initialize_for_time_warping(L, degree)

# Training parameters
num_epochs = 1000

# Training hyperparameters
tau_psi = 5000
tau_beta = 200

losses = []

alpha_loss_increase = []
gamma_loss_increase = []
chi_loss_increase = []
d_loss_increase = []

alpha_learning_rate = []
gamma_learning_rate = []
chi_learning_rate = []
d_learning_rate = []

alpha_iters = []
gamma_iters = []
chi_iters = []
d_iters = []

total_time = 0
epoch_time = 0
for epoch in range(num_epochs):
    start_time = time.time()  # Record the start time of the epoch

    result = model.log_obj_with_backtracking_line_search_and_time_warping(tau_psi, tau_beta)
    loss = result["loss"]
    psi_penalty = result["psi_penalty"]
    beta_penalty = result["beta_penalty"]
    dalpha = result["dlogL_dalpha"]
    dgamma = result["dlogL_dgamma"]
    dchi = result["dlogL_dchi"]
    dd = result["dlogL_dd"]

    losses.append(loss)

    alpha_loss_increase.append(result["alpha_loss_increase"])
    gamma_loss_increase.append(result["gamma_loss_increase"])
    chi_loss_increase.append(result["chi_loss_increase"])
    d_loss_increase.append(result["d_loss_increase"])

    alpha_learning_rate.append(result["smooth_alpha"])
    gamma_learning_rate.append(result["smooth_gamma"])
    chi_learning_rate.append(result["smooth_chi"])
    d_learning_rate.append(result["smooth_d"])

    alpha_iters.append(result["iters_alpha"])
    gamma_iters.append(result["iters_gamma"])
    chi_iters.append(result["iters_chi"])
    d_iters.append(result["iters_d"])

    end_time = time.time()  # Record the end time of the epoch
    elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
    epoch_time += elapsed_time  # Record the elapsed time for the epoch
    total_time += elapsed_time  # Calculate the total time for training

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss}, Epoch Time: {epoch_time/60:.2f} mins, Total Time: {total_time/(60*60):.2f} hrs")
        epoch_time = 0  # Reset the epoch time

num_epochs = len(losses)
losses = np.array(losses)
alpha_learning_rate = np.array(alpha_learning_rate)
gamma_learning_rate = np.array(gamma_learning_rate)
chi_learning_rate = np.array(chi_learning_rate)
d_learning_rate = np.array(d_learning_rate)
alpha_iters = np.array(alpha_iters)
gamma_iters = np.array(gamma_iters)
chi_iters = np.array(chi_iters)
d_iters = np.array(d_iters)
plt.plot(np.arange(0, num_epochs), losses[0:])
plt.title('Losses')
plt.show()
plt.plot(np.arange(0, num_epochs), alpha_learning_rate)
plt.title('Alpha Learning Rates')
plt.show()
plt.plot(np.arange(0, num_epochs), gamma_learning_rate)
plt.title('Beta Learning Rates')
plt.show()
plt.plot(np.arange(0, num_epochs), chi_learning_rate)
plt.title('G Learning Rates')
plt.show()
plt.plot(np.arange(0, num_epochs), d_learning_rate)
plt.title('d Learning Rates')
plt.show()
plt.plot(np.arange(0, num_epochs), alpha_iters)
plt.title('Alpha Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), gamma_iters)
plt.title('Beta Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), chi_iters)
plt.title('G Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), d_iters)
plt.title('d Iters')
plt.show()

exp_alpha_c = (np.exp(model.alpha) @ model.alpha_prime_multiply) + model.alpha_prime_add
psi = exp_alpha_c @ model.U_psi
psi_norm = (1 / (psi[:, (model.V.shape[0]-1), np.newaxis])) * psi
time_matrix = max(model.time) * psi_norm @ model.V
B_sparse = [BSpline.design_matrix(time, model.knots, model.degree).transpose() for time in time_matrix]
beta = np.exp(model.gamma)
exp_chi = np.exp(model.chi)
G = (1/np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi
GBeta = G @ beta
GBetaBPsi = np.vstack([GBeta[i] @ b for i, b in enumerate(B_sparse)])
diagdJ_plus_GBetaB = model.d + GBetaBPsi  # variable
lambda_manual = np.exp(diagdJ_plus_GBetaB)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(stim_time, avg_lambda_manual)
plt.show()
np.mean(np.square(intensity - lambda_manual))
for i in range(model.Y.shape[0]):
    plt.plot(stim_time, lambda_manual[i, :] + i * 2)
plt.show()

latent_factors_manual = beta @ model.V
for i in range(L):
    # plt.plot(np.concatenate([[stim_time[0] - 0.02, stim_time[0] - 0.01], stim_time]), model.beta[i, :])
    plt.plot(stim_time, latent_factors_manual[i, :])
    plt.title(f'Factor [{i}, :]')
plt.show()

for i in range(model.Y.shape[0]):
    plt.plot(stim_time, time_matrix[i, :] + i * 0.01)
plt.show()

G_and_d = np.concatenate([G, model.d], axis=1)
