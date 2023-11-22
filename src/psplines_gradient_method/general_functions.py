import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from scipy.interpolate import BSpline
import json
import re


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


def plot_outputs(model, data, output_dir, epoch, batch=10, time_warping=False):
    R, Q = model.zeta.shape
    K, L = model.chi.shape
    stim_time = model.time
    T = stim_time.shape[0]
    objects = model.compute_loss_objects(0, 0, 0, time_warping)
    time_matrix = objects["time_matrix"]
    psi_norm = objects["psi_norm"]
    kappa_norm = objects["kappa_norm"]
    B_sparse = objects["B_sparse"]
    exp_chi = np.exp(model.chi)  # variable
    G = (1 / np.sum(exp_chi, axis=1).reshape(-1, 1)) * exp_chi  # variable
    beta = np.exp(model.gamma)
    E = np.exp(model.c)  # variable
    E_beta_Bpsi = [E[k][:, np.newaxis] * beta @ b for k, b in enumerate(B_sparse)]
    GEBetaBPsi = np.vstack([G[k] @ e for k, e in enumerate(E_beta_Bpsi)])
    avg_lambda_intensities = np.mean(GEBetaBPsi, axis=0)
    if not time_warping:
        R = 1
    plt.figure()
    for i in range(R):
        plt.plot(stim_time, avg_lambda_intensities[i * T:(i + 1) * T] + i * 0.01)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, f'main_AvgLambdaIntensities.png'))
    latent_factors = beta @ model.V
    plt.figure()
    for i in range(L):
        # plt.plot(np.concatenate([[stim_time[0] - 0.02, stim_time[0] - 0.01], stim_time]), model.beta[i, :])
        plt.plot(stim_time, latent_factors[i, :], label=f'Factor [{i}, :]')
        plt.title(f'Factors')
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, f'main_LatentFactors_{epoch}.png'))
    time_matrix_psi = max(model.time) * (psi_norm @ model.V)
    for i in range(0, model.Y.shape[0], batch):
        this_batch = batch if i + batch < model.Y.shape[0] else model.Y.shape[0] - i
        plt.figure()
        for j in range(this_batch):
            plt.plot(stim_time, time_matrix_psi[i + j, :] + i * 0.05)
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(output_dir, f'main_TimeMatrixPsi_batch{i}.png'))
    time_matrix_kappa = max(model.time) * (kappa_norm @ model.V)
    plt.figure()
    for i in range(R):
        plt.plot(stim_time, time_matrix_kappa[i, :] + i * 0.1)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, f'main_TimeMatrixKappa.png'))
    B_sparse_kappa = [BSpline.design_matrix(time, model.knots, model.degree).transpose() for time in time_matrix_kappa]
    latent_factors_kappa = [beta @ b for b in B_sparse_kappa]
    global_max = np.max(GEBetaBPsi)
    upper_limit = global_max * 1.1
    for r in range(R):
        for i in range(0, model.Y.shape[0], batch):
            this_batch = batch if i + batch < model.Y.shape[0] else model.Y.shape[0] - i

            plt.figure()
            for j in range(this_batch):
                plt.plot(stim_time, time_matrix[i + j, r * T:(r + 1) * T] + i * 0.01)
            plt.ylim(bottom=0)
            plt.savefig(os.path.join(output_dir, f'main_TimeMatrix_Trial{r}_batch{i}.png'))

            plt.figure(figsize=(10, 10))
            sorted_indices = sorted(range(this_batch), key=lambda j: np.argmax(G[i + j]), reverse=True)
            for k, j in enumerate(sorted_indices):
                plt.subplot(2, 1, 1)
                plt.plot(stim_time, GEBetaBPsi[i + j, r * T:(r + 1) * T] + k * 0.0,
                         label=f'I={i + j}, C={np.argmax(G[i + j])}, V={round(np.max(G[i + j]), 2)}')
                plt.ylim(bottom=0, top=upper_limit)
                plt.subplot(2, 1, 2)
                plt.plot(stim_time, data.intensity[i + j, :stim_time.shape[0]] + k * 1, label=f'I={i + j}')
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


def load_likelihoods(K, R, L, intensity_mltply, intensity_bias, data_seed, param_seed=None):
    base_dir = os.path.join(os.getcwd(), 'outputs')
    if param_seed is None:
        param_seed = r'\d+'
    regex_pattern = fr"paramSeed({param_seed})_dataSeed{data_seed}_L{L}_K{K}_R{R}_int.mltply{intensity_mltply}_int.add{intensity_bias}_tauBeta\d+_tauS\d+_iters\d+_betaFirst\d+_notes-.+"
    pattern = re.compile(regex_pattern)
    file_name = 'log_likelihoods.json'
    all_data = []
    all_param_seeds = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        match = pattern.match(entry)
        if os.path.isdir(full_path) and match:
            json_path = os.path.join(full_path, file_name)
            if os.path.exists(json_path):
                print(f"Found '{file_name}' in '{full_path}'")
                with open(json_path, 'r') as file:
                    data = json.load(file)
                    all_data.append(data)
                all_param_seeds.append(match.group(1))
    print(f"Found {len(all_data)} file(s)")

    return all_data, all_param_seeds, base_dir


def plot_likelihoods(true_data, K, R, L, intensity_mltply, intensity_bias, data_seed):
    true_likelihood = true_data.likelihood()
    print(f"True likelihood: {true_likelihood}")
    all_data, all_param_seeds, output_dir = load_likelihoods(K, R, L, intensity_mltply, intensity_bias, data_seed)
    ground_truth_start = load_likelihoods(K, R, L, intensity_mltply, intensity_bias, data_seed, 'TRUTH')[0][0]
    max_length = max(len(data) for data in all_data)
    true_likelihood_vector = [true_likelihood] * max_length
    plt.figure(figsize=(10, 6))
    plt.plot(true_likelihood_vector, label='True Likelihood')
    plt.plot(ground_truth_start, label='Ground Truth Start')
    for i, data in enumerate(all_data):
        plt.plot(data, label=f'paramSeed {all_param_seeds[i]}', alpha=0.7)

    plt.xlabel('Iterations')
    plt.ylabel('Likelihood')
    plt.title('Plot of likelihood values')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'LikelihoodTrajectories_dataSeed{data_seed}.png'))


def write_outputs(output_dir, log_likelihoods, is_empty, model, output_str):
    with open(os.path.join(output_dir, 'log.txt'), 'a') as file:
        file.write(output_str)
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(os.path.join(output_dir, 'log_likelihoods.json'), 'r+b') as file:
        _ = file.seek(-1, 2)  # Go to the one character before the end of the file
        if file.read(1) != b']':
            raise ValueError("JSON file must end with a ']'")
        _ = file.seek(-1, 2)  # Go back to the position just before the ']'
        for item in log_likelihoods:
            if not is_empty:
                _ = file.write(b',' + json.dumps(item).encode('utf-8'))
            else:
                _ = file.write(json.dumps(item).encode('utf-8'))
        _ = file.write(b']')


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value