import os
import numpy as np
from scipy.interpolate import BSpline
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel
from src.psplines_gradient_method.general_functions import plot_binned, plot_spikes
import matplotlib.pyplot as plt
import time

K = 100
R = 15
L = 3
intensity_mltply = 25
intensity_bias = 0.1
folder_name = f'main_L{L}_K{K}_R{R}_int.mltply{intensity_mltply}_int.add{intensity_bias}_NoTimeWarp'
output_dir = os.path.join(os.getcwd(), 'outputs', folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

self = DataAnalyzer().initialize(K=K, R=R, intensity_mltply=intensity_mltply, intensity_bias=intensity_bias, max_offset=0)
binned, stim_time = self.sample_data()
intensity, latent_factors = self.intensity, self.latent_factors
binned_spikes = np.where(binned >= 1)
plot_binned(binned, output_dir)
plot_spikes(binned_spikes, R, output_dir)
self.plot_intensity_and_latents(output_dir)

Y = binned  # K x T
degree = 3
#L = self.latent_factors.shape[0] #- 1
model = SpikeTrainModel(Y, stim_time).initialize_for_time_warping(L, degree)

# Training parameters
num_epochs = 1000

# Training hyperparameters
tau_psi = 10000
tau_beta = 8000

likelihoods = []

alpha_loss_increase = []
gamma_loss_increase = []
zeta_loss_increase = []
chi_loss_increase = []

alpha_learning_rate = []
gamma_learning_rate = []
zeta_learning_rate = []
chi_learning_rate = []

alpha_iters = []
gamma_iters = []
zeta_iters = []
chi_iters = []

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

    likelihoods.append(likelihood)

    alpha_loss_increase.append(result["alpha_loss_increase"])
    gamma_loss_increase.append(result["gamma_loss_increase"])
    zeta_loss_increase.append(result["zeta_loss_increase"])
    chi_loss_increase.append(result["chi_loss_increase"])

    alpha_learning_rate.append(result["smooth_alpha"])
    gamma_learning_rate.append(result["smooth_gamma"])
    zeta_learning_rate.append(result["smooth_zeta"])
    chi_learning_rate.append(result["smooth_chi"])

    alpha_iters.append(result["iters_alpha"])
    gamma_iters.append(result["iters_gamma"])
    zeta_iters.append(result["iters_zeta"])
    chi_iters.append(result["iters_chi"])

    end_time = time.time()  # Record the end time of the epoch
    elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
    epoch_time += elapsed_time  # Record the elapsed time for the epoch
    total_time += elapsed_time  # Calculate the total time for training

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Likelihood {likelihood}, Epoch Time: {epoch_time/60:.2f} mins, Total Time: {total_time/(60*60):.2f} hrs")
        epoch_time = 0  # Reset the epoch time

R, Q = model.zeta.shape
K = model.Y.shape[0]
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
beta[(L-1), :] = 1
exp_chi = np.exp(model.chi)
exp_chi[:, 0] = 1
G = (1/np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi
GBeta = G @ beta
GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])
avg_lambda_intensities = np.mean(GBetaBPsi, axis=0)
batch = 10

plt.figure()
for i in range(R):
    plt.plot(stim_time, avg_lambda_intensities[i*T:(i+1)*T] + i * 0.01)
plt.savefig(os.path.join(output_dir, f'main_AvgLambdaIntensities.png'))

latent_factors = beta @ model.V
plt.figure()
for i in range(L):
    # plt.plot(np.concatenate([[stim_time[0] - 0.02, stim_time[0] - 0.01], stim_time]), model.beta[i, :])
    plt.plot(stim_time, latent_factors[i, :], label=f'Factor [{i}, :]')
    plt.title(f'Factors')
plt.savefig(os.path.join(output_dir, f'main_LatentFactors.png'))

time_matrix_psi = max(model.time) * (psi_norm @ model.V)
for i in range(0, model.Y.shape[0], batch):
    this_batch = batch if i + batch < model.Y.shape[0] else model.Y.shape[0] - i
    plt.figure()
    for j in range(this_batch):
        plt.plot(stim_time, time_matrix_psi[i+j, :] + i * 0.05)
    plt.savefig(os.path.join(output_dir, f'main_TimeMatrixPsi_batch{i}.png'))

time_matrix_kappa = max(model.time) * (kappa_norm @ model.V)
plt.figure()
for i in range(R):
    plt.plot(stim_time, time_matrix_kappa[i, :] + i * 0.1)
plt.savefig(os.path.join(output_dir, f'main_TimeMatrixKappa.png'))

B_sparse_kappa = [BSpline.design_matrix(time, model.knots, model.degree).transpose() for time in time_matrix_kappa]
latent_factors_kappa = [beta @ b for b in B_sparse_kappa]
for r in range(1):
    for i in range(0, model.Y.shape[0], batch):
        this_batch = batch if i + batch < model.Y.shape[0] else model.Y.shape[0] - i

        plt.figure()
        for j in range(this_batch):
            plt.plot(stim_time, time_matrix[i+j, r*T:(r+1)*T] + i * 0.01)
        plt.savefig(os.path.join(output_dir, f'main_TimeMatrix_Trial{r}_batch{i}.png'))

        plt.figure(figsize=(10, 10))
        sorted_indices = sorted(range(this_batch), key=lambda j: np.argmax(G[i+j]), reverse=True)
        for k, j in enumerate(sorted_indices):
            plt.subplot(2, 1, 1)
            plt.plot(stim_time, GBetaBPsi[i+j, r*T:(r+1)*T] + k*0.0,
                     label=f'I={i+j}, C={np.argmax(G[i+j])}, V={round(np.max(G[i+j]),2)}')
            plt.subplot(2, 1, 2)
            plt.plot(stim_time, intensity[i+j, :] + k * 0.1, label=f'I={i + j}')
        plt.subplot(2, 1, 1)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'main_LambdaIntensities_Trial{r}_batch{i}.png'))

    plt.figure()
    for i in range(L):
        plt.plot(stim_time, latent_factors_kappa[r][i, :])
        plt.title(f'Factor [{i}, :]')
    plt.savefig(os.path.join(output_dir, f'main_LatentFactorsKappa_Trial{r}.png'))

plt.close()
plt.show()
G_and_d = np.concatenate([G, model.d], axis=1)

# Debug plots
num_epochs = len(likelihoods)
likelihoods = np.array(likelihoods)
alpha_learning_rate = np.array(alpha_learning_rate)
gamma_learning_rate = np.array(gamma_learning_rate)
chi_learning_rate = np.array(chi_learning_rate)
alpha_iters = np.array(alpha_iters)
gamma_iters = np.array(gamma_iters)
chi_iters = np.array(chi_iters)
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
plt.plot(np.arange(0, num_epochs), alpha_iters)
plt.title('Alpha Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), gamma_iters)
plt.title('Beta Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), chi_iters)
plt.title('G Iters')
plt.show()
