import numpy as np
from scipy.interpolate import BSpline
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel
from src.psplines_gradient_method.general_functions import plot_binned, plot_spikes
import matplotlib.pyplot as plt
import time

self = DataAnalyzer().initialize(K=100, R=3, max_offset=0)
binned, stim_time = self.sample_data()
binned_spikes = np.where(binned >= 1)
# plot_spikes(binned_spikes)
intensity, latent_factors = self.intensity, self.latent_factors
# plot_binned(binned)
# self.plot_intensity_and_latents()

Y = binned  # K x T
degree = 3
L = self.latent_factors.shape[0] - 1
model = SpikeTrainModel(Y, stim_time).initialize_for_time_warping(L, degree)

# Training parameters
num_epochs = 1000

# Training hyperparameters
tau_psi = 5000
tau_beta = 800

likelihoods = []

alpha_loss_increase = []
gamma_loss_increase = []
zeta_loss_increase = []
chi_loss_increase = []
d_loss_increase = []

alpha_learning_rate = []
gamma_learning_rate = []
zeta_learning_rate = []
chi_learning_rate = []
d_learning_rate = []

alpha_iters = []
gamma_iters = []
zeta_iters = []
chi_iters = []
d_iters = []

total_time = 0
epoch_time = 0
for epoch in range(num_epochs):
    start_time = time.time()  # Record the start time of the epoch

    result = model.log_obj_with_backtracking_line_search_and_time_warping(tau_psi, tau_beta)
    likelihood = result["likelihood"]
    psi_penalty = result["psi_penalty"]
    kappa_penalty = result["kappa_penalty"]
    beta_penalty = result["beta_penalty"]
    dalpha = result["dlogL_dalpha"]
    dgamma = result["dlogL_dgamma"]
    dzeta = result["dlogL_dzeta"]
    dchi = result["dlogL_dchi"]
    dd = result["dlogL_dd"]

    likelihoods.append(likelihood)

    alpha_loss_increase.append(result["alpha_loss_increase"])
    gamma_loss_increase.append(result["gamma_loss_increase"])
    zeta_loss_increase.append(result["zeta_loss_increase"])
    chi_loss_increase.append(result["chi_loss_increase"])
    d_loss_increase.append(result["d_loss_increase"])

    alpha_learning_rate.append(result["smooth_alpha"])
    gamma_learning_rate.append(result["smooth_gamma"])
    zeta_learning_rate.append(result["smooth_zeta"])
    chi_learning_rate.append(result["smooth_chi"])
    d_learning_rate.append(result["smooth_d"])

    alpha_iters.append(result["iters_alpha"])
    gamma_iters.append(result["iters_gamma"])
    zeta_iters.append(result["iters_zeta"])
    chi_iters.append(result["iters_chi"])
    d_iters.append(result["iters_d"])

    end_time = time.time()  # Record the end time of the epoch
    elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
    epoch_time += elapsed_time  # Record the elapsed time for the epoch
    total_time += elapsed_time  # Calculate the total time for training

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Likelihood {likelihood}, Epoch Time: {epoch_time/60:.2f} mins, Total Time: {total_time/(60*60):.2f} hrs")
        epoch_time = 0  # Reset the epoch time

num_epochs = len(likelihoods)
likelihoods = np.array(likelihoods)
alpha_learning_rate = np.array(alpha_learning_rate)
gamma_learning_rate = np.array(gamma_learning_rate)
chi_learning_rate = np.array(chi_learning_rate)
d_learning_rate = np.array(d_learning_rate)
alpha_iters = np.array(alpha_iters)
gamma_iters = np.array(gamma_iters)
chi_iters = np.array(chi_iters)
d_iters = np.array(d_iters)
plt.plot(np.arange(0, num_epochs), likelihoods[0:])
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

R, Q = model.zeta.shape
K = model.d.shape[0]
T = model.time.shape[0]
exp_alpha_c = (np.exp(model.alpha) @ model.alpha_prime_multiply) + np.repeat(model.alpha_prime_add, K, axis=0)
psi = exp_alpha_c @ model.U_ones  # variable
psi_norm = (1 / (psi[:, (Q - 1), np.newaxis])) * psi  # variable, called \psi' in the document
exp_zeta_c = (np.exp(model.zeta) @ model.alpha_prime_multiply) + np.repeat(model.alpha_prime_add, R, axis=0)
kappa = exp_zeta_c @ model.U_ones  # variable
kappa_norm = (1 / (kappa[:, (Q - 1), np.newaxis])) * kappa  # variable, called \kappa' in the document
time_matrix = 0.5 * max(model.time) * np.hstack([(psi_norm + kappa_norm[r]) @ model.V for r in range(R)])  # variable
B_sparse = [BSpline.design_matrix(time, model.knots, model.degree).transpose() for time in time_matrix]
beta = np.exp(model.gamma)
exp_chi = np.exp(model.chi)
G = (1/np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi
GBeta = G @ beta
GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])
diagdJ_plus_GBetaB = model.d + GBetaBPsi  # variable
lambda_manual = np.exp(diagdJ_plus_GBetaB)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
for i in range(R):
    plt.plot(stim_time, avg_lambda_manual[i*T:(i+1)*T] + i * 2)
plt.show()

latent_factors_manual = beta @ model.V
for i in range(L):
    # plt.plot(np.concatenate([[stim_time[0] - 0.02, stim_time[0] - 0.01], stim_time]), model.beta[i, :])
    plt.plot(stim_time, latent_factors_manual[i, :])
    plt.title(f'Factor [{i}, :]')
plt.show()

time_matrix_psi = max(model.time) * (psi_norm @ model.V)
for i in range(model.Y.shape[0]):
    plt.plot(stim_time, time_matrix_psi[i, :] + i * 0.05)
plt.show()
time_matrix_kappa = max(model.time) * (kappa_norm @ model.V)
for i in range(R):
    plt.plot(stim_time, time_matrix_kappa[i, :] + i * 0.1)
plt.show()

B_sparse_kappa = [BSpline.design_matrix(time, model.knots, model.degree).transpose() for time in time_matrix_kappa]
latent_factors_kappa = [beta @ b for b in B_sparse_kappa]
r = 0
for i in range(model.Y.shape[0]):
    plt.plot(stim_time, time_matrix[i, r*T:(r+1)*T] + i * 0.01)
plt.show()
for i in range(model.Y.shape[0]):
    plt.plot(stim_time, lambda_manual[i, r*T:(r+1)*T] + i * 2)
plt.show()
for i in range(L):
    plt.plot(stim_time, latent_factors_kappa[r][i, :])
    plt.title(f'Factor [{i}, :]')
plt.show()

G_and_d = np.concatenate([G, model.d], axis=1)
