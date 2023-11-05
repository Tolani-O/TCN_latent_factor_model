import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from scipy.interpolate import BSpline


def create_precision_matrix(P):
    Omega = np.zeros((P, P))
    # fill the main diagonal with 2s
    np.fill_diagonal(Omega, 2)
    # fill the subdiagonal and superdiagonal with -1s
    np.fill_diagonal(Omega[1:], -1)
    np.fill_diagonal(Omega[:, 1:], -1)
    # set the last element to 1
    Omega[-1, -1] = 1
    return Omega


def create_first_diff_matrix(P):
    D = np.zeros((P-1, P))
    # fill the main diagonal with -1s
    np.fill_diagonal(D, -1)
    # fill the superdiagonal with 1s
    np.fill_diagonal(D[:, 1:], 1)
    return D


def create_second_diff_matrix(P):
    D = np.zeros((P-2, P))
    # fill the main diagonal with 1s
    np.fill_diagonal(D, 1)
    # fill the subdiagonal and superdiagonal with -2s
    np.fill_diagonal(D[:, 2:], 1)
    np.fill_diagonal(D[:, 1:], -2)
    # set the last element to 1
    D[-1, -1] = 1
    return D


def plot_spikes(binned, output_dir, x_offset=0):
    # Group entries by unique values of s[0]
    spikes = np.where(binned >= 1)
    unique_s_0 = np.unique(spikes[0])
    grouped_s = []
    for i in unique_s_0:
        indices = np.where(spikes[0] == i)[0]
        values = (spikes[1][indices] - x_offset)/1000
        grouped_s.append((i, values))
    aspect_ratio = binned.shape[0] / binned.shape[1]
    w, h = figaspect(aspect_ratio)
    plt.figure(figsize=(w, h))
    for group in grouped_s:
        plt.scatter(group[1], np.zeros_like(group[1]) + group[0], s=1, c='black')
    plt.savefig(os.path.join(output_dir, 'groundTruth_spikes.png'))


def plot_binned(binned, output_dir):
    # plot binned spikes
    _, ax = plt.subplots()
    ax.imshow(binned)
    ax.invert_yaxis()
    plt.savefig(os.path.join(output_dir, 'groundTruth_binned.png'))


def plot_intensity_and_latents(time, latent_factors, intensity, output_dir):
    # plot latent factors
    plt.figure()
    for i in range(latent_factors.shape[0]):
        plt.plot(time, latent_factors[i, :] + i)
    plt.savefig(os.path.join(output_dir, 'groundTruth_latent_factors.png'))

    # plot neuron intensities
    plt.figure()
    for i in range(intensity.shape[0]):
        plt.plot(time, intensity[i, :time.shape[0]] + i * 1)
    plt.savefig(os.path.join(output_dir, 'groundTruth_intensities.png'))


def plot_bsplines(B, time, output_dir):
    # plot bsplines
    start = 190
    num_basis = 10
    plt.figure()
    for i in range(num_basis):
        plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
    plt.savefig(os.path.join(output_dir, 'groundTruth_bsplines.png'))


def plot_outputs(model, data, output_dir, batch=10, time_warping=False):
    R, Q = model.zeta.shape
    K, L = model.chi.shape
    stim_time = model.time
    T = stim_time.shape[0]
    objects = model.compute_loss_objects(K, L, Q, R, 0, 0, 0, time_warping)
    time_matrix = objects["time_matrix"]
    psi_norm = objects["psi_norm"]
    kappa_norm = objects["kappa_norm"]
    B_sparse = objects["B_sparse"]
    exp_chi = np.exp(model.chi)  # variable
    G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
    beta = np.exp(model.gamma)
    GBeta = G @ beta  # didnt change
    GBetaBPsi = np.vstack([GBeta[k] @ b for k, b in enumerate(B_sparse)])
    avg_lambda_intensities = np.mean(GBetaBPsi, axis=0)
    if not time_warping:
        R = 1
    plt.figure()
    for i in range(R):
        plt.plot(stim_time, avg_lambda_intensities[i * T:(i + 1) * T] + i * 0.01)
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
            plt.plot(stim_time, time_matrix_psi[i + j, :] + i * 0.05)
        plt.savefig(os.path.join(output_dir, f'main_TimeMatrixPsi_batch{i}.png'))
    time_matrix_kappa = max(model.time) * (kappa_norm @ model.V)
    plt.figure()
    for i in range(R):
        plt.plot(stim_time, time_matrix_kappa[i, :] + i * 0.1)
    plt.savefig(os.path.join(output_dir, f'main_TimeMatrixKappa.png'))
    B_sparse_kappa = [BSpline.design_matrix(time, model.knots, model.degree).transpose() for time in time_matrix_kappa]
    latent_factors_kappa = [beta @ b for b in B_sparse_kappa]
    for r in range(R):
        for i in range(0, model.Y.shape[0], batch):
            this_batch = batch if i + batch < model.Y.shape[0] else model.Y.shape[0] - i

            plt.figure()
            for j in range(this_batch):
                plt.plot(stim_time, time_matrix[i + j, r * T:(r + 1) * T] + i * 0.01)
            plt.savefig(os.path.join(output_dir, f'main_TimeMatrix_Trial{r}_batch{i}.png'))

            plt.figure(figsize=(10, 10))
            sorted_indices = sorted(range(this_batch), key=lambda j: np.argmax(G[i + j]), reverse=True)
            for k, j in enumerate(sorted_indices):
                plt.subplot(2, 1, 1)
                plt.plot(stim_time, GBetaBPsi[i + j, r * T:(r + 1) * T] + k * 0.0,
                         label=f'I={i + j}, C={np.argmax(G[i + j])}, V={round(np.max(G[i + j]), 2)}')
                plt.subplot(2, 1, 2)
                plt.plot(stim_time, data.intensity[i + j, :] + k * 0.1, label=f'I={i + j}')
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


def plot_training_metrics():
    global num_epochs, likelihoods, alpha_learning_rate, gamma_learning_rate, chi_learning_rate, alpha_iters, gamma_iters, chi_iters
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
