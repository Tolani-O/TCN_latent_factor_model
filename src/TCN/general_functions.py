import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.figure import figaspect
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


def plot_latent_coupling(latent_coupling, output_dir):
    # plot latent couplings
    plt.figure(figsize=(10, 10))
    sns.heatmap(latent_coupling, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Heatmap of clusters')
    plt.savefig(os.path.join(output_dir, f'groundTruth_neuron_clusters.png'))


def plot_bsplines(B, time, output_dir):
    # plot bsplines
    start = 190
    num_basis = 10
    plt.figure()
    for i in range(num_basis):
        plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
    plt.savefig(os.path.join(output_dir, 'groundTruth_bsplines.png'))


def plot_outputs(model, binned, true_intensity, stim_time, Bspline_matrix, output_dir, name, epoch, batch=10):

    output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.eval()
    with torch.no_grad():
        binned = torch.stack(binned)
        latent_coeffs, cluster_attn, firing_attn = model(binned)
        latent_factors = torch.matmul(latent_coeffs, Bspline_matrix).numpy()
        cluster_attn = cluster_attn.numpy()
        firing_attn = firing_attn.numpy()

    plt.figure(figsize=(10,10))
    sns.heatmap(cluster_attn, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Heatmap of cluster_attn')
    plt.savefig(os.path.join(output_dir, f'cluster_attn_{epoch}.png'))

    plt.figure(figsize=(10,10))
    sns.heatmap(firing_attn, annot=True, fmt=".2f", annot_kws={"color": "blue"})
    plt.title('Heatmap of firing_attn')
    plt.savefig(os.path.join(output_dir, f'firing_attn_{epoch}.png'))

    L, T = latent_factors.shape
    K = cluster_attn.shape[0]
    if K < batch:
        batch = cluster_attn.shape[0]

    weighted_firing = cluster_attn * firing_attn
    learned_intensity = weighted_firing @ latent_factors
    avg_lambda_intensities = np.mean(learned_intensity, axis=0)

    plt.figure()
    plt.plot(stim_time, avg_lambda_intensities)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, 'main_AvgLambdaIntensities.png'))

    plt.figure()
    for i in range(L):
        plt.plot(stim_time, latent_factors[i, :], label=f'Factor [{i}, :]')
        plt.title(f'Factors')
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, f'main_LatentFactors_{epoch}.png'))

    global_max = np.max(learned_intensity)
    upper_limit = global_max * 1.1
    for i in range(0, K, batch):
        this_batch = batch if i + batch < K else K - i

        plt.figure(figsize=(10, 10))
        sorted_indices = sorted(range(this_batch), key=lambda j: np.argmax(cluster_attn[i + j]), reverse=True)
        for k, j in enumerate(sorted_indices):
            plt.subplot(2, 1, 1)
            plt.plot(stim_time, learned_intensity[i + j, :] + k * 0.01,
                     label=f'I={i + j}, C={np.argmax(cluster_attn[i + j])}, V={round(np.max(cluster_attn[i + j]), 2)}')
            plt.ylim(bottom=0, top=upper_limit)
            plt.subplot(2, 1, 2)
            plt.plot(stim_time, true_intensity[i + j, :stim_time.shape[0]] + k * 1, label=f'I={i + j}')
        plt.subplot(2, 1, 1)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'main_LambdaIntensities_Trial_batch{i}.png'))


def plot_likelihoods(true_likelihood, output_dir, name, metric, cutoff):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    else:
        file_name = 'losses'
    file_name = f'{file_name}_{name.lower()}.json'
    if name.lower()=='test':
        folder = 'Test'
    else:
        folder = 'Train'
    json_path = os.path.join(output_dir, file_name)
    with open(json_path, 'r') as file:
        metric_data = json.load(file)
    metric_data = metric_data[cutoff:]
    plt.figure(figsize=(10, 6))
    plt.plot(metric_data, label=metric)
    if 'likelihood' in metric.lower():
        true_likelihood_vector = [true_likelihood] * len(metric_data)
        plt.plot(true_likelihood_vector, label='True Log Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel(metric)
    plt.title('Plot of metric values')
    plt.legend()
    plt.savefig(os.path.join(output_dir, folder, f'{metric}_{name}_Trajectories.png'))


def write_log_and_model(model,loss_function, output_str, output_dir, epoch):
    with open(os.path.join(output_dir, 'log.txt'), 'a') as file:
        file.write(output_str)
    models_path = os.path.join(output_dir, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    torch.save(model, os.path.join(models_path, f'model_{epoch}.pth'))
    torch.save(loss_function, os.path.join(models_path, f'loss_{epoch}.pth'))


def write_outputs(list, name, metric, output_dir, starts_out_empty):
    if 'likelihood' in metric.lower():
        file_name = 'log_likelihoods'
    else:
        file_name = 'losses'
    file_name = f'{file_name}_{name.lower()}.json'
    with open(os.path.join(output_dir, file_name), 'r+b') as file:
        _ = file.seek(-1, 2)  # Go to the one character before the end of the file
        if file.read(1) != b']':
            raise ValueError("JSON file must end with a ']'")
        _ = file.seek(-1, 2)  # Go back to the position just before the ']'
        currently_empty = starts_out_empty
        for item in list:
            if not currently_empty:
                _ = file.write(b',' + json.dumps(item).encode('utf-8'))
            else:
                _ = file.write(json.dumps(item).encode('utf-8'))
                currently_empty = 0
        _ = file.write(b']')


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value