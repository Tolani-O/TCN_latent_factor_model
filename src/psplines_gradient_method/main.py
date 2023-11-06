import os
import sys
sys.path.append(os.path.abspath('.'))

import pickle
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel
from src.psplines_gradient_method.general_functions import plot_outputs, plot_spikes, plot_intensity_and_latents
import time
import argparse


def main(K, R, L, intensity_mltply, intensity_bias, tau_psi, tau_beta, tau_s, beta_first, notes, num_epochs):
    # K = 100
    # R = 15
    # L = 3
    # intensity_mltply = 25
    # intensity_bias = 0.1
    # # Training hyperparameters
    # tau_psi = 10000
    # tau_beta = 8000
    # num_epochs = 1000

    folder_name = (f'main_L{L}_K{K}_R{R}_int.mltply{intensity_mltply}_int.add{intensity_bias}'
                   f'_tauBeta{tau_beta}_tauS{tau_s}_iters{num_epochs}_betaFirst{beta_first}'
                   f'_notes-{notes}_reparam')
    print(f'folder_name: {folder_name}')
    output_dir = os.path.join(os.getcwd(), 'outputs', folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'log.txt'), 'w'):
        pass

    data = DataAnalyzer().initialize(K=K, R=R, intensity_mltply=intensity_mltply, intensity_bias=intensity_bias, max_offset=0)
    binned, stim_time = data.sample_data()
    true_likelihood, beta_s1_penalty, beta_s2_penalty = data.likelihood(tau_beta)
    plot_spikes(binned, output_dir)
    plot_intensity_and_latents(data.time, data.latent_factors, data.intensity, output_dir)
    Y = binned  # K x T
    degree = 3
    # L = self.latent_factors.shape[0] #- 1
    model = SpikeTrainModel(Y, stim_time).initialize_for_time_warping(L, degree)

    # Training parameters

    likelihoods = []
    alpha_loss_increase = []
    gamma_loss_increase = []
    d1_loss_increase = []
    d2_loss_increase = []
    zeta_loss_increase = []
    chi_loss_increase = []
    alpha_learning_rate = []
    gamma_learning_rate = []
    d1_learning_rate = []
    d2_learning_rate = []
    zeta_learning_rate = []
    chi_learning_rate = []
    alpha_iters = []
    gamma_iters = []
    d1_iters = []
    d2_iters = []
    zeta_iters = []
    chi_iters = []
    total_time = 0
    epoch_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()  # Record the start time of the epoch

        result = model.log_obj_with_backtracking_line_search_and_time_warping(tau_psi, tau_beta, tau_s, beta_first)
        beta_first = 1 - beta_first
        likelihood = result["likelihood"]
        likelihoods.append(likelihood)

        alpha_loss_increase.append(result["alpha_loss_increase"])
        gamma_loss_increase.append(result["gamma_loss_increase"])
        d1_loss_increase.append(result["d1_loss_increase"])
        d2_loss_increase.append(result["d2_loss_increase"])
        zeta_loss_increase.append(result["zeta_loss_increase"])
        chi_loss_increase.append(result["chi_loss_increase"])

        alpha_learning_rate.append(result["smooth_alpha"])
        gamma_learning_rate.append(result["smooth_gamma"])
        d1_learning_rate.append(result["smooth_d1"])
        d2_learning_rate.append(result["smooth_d2"])
        zeta_learning_rate.append(result["smooth_zeta"])
        chi_learning_rate.append(result["smooth_chi"])

        alpha_iters.append(result["iters_alpha"])
        gamma_iters.append(result["iters_gamma"])
        d1_iters.append(result["iters_d1"])
        d2_iters.append(result["iters_d2"])
        zeta_iters.append(result["iters_zeta"])
        chi_iters.append(result["iters_chi"])

        end_time = time.time()  # Record the end time of the epoch
        elapsed_time = end_time - start_time  # Calculate the elapsed time for the epoch
        epoch_time += elapsed_time  # Record the elapsed time for the epoch
        total_time += elapsed_time  # Calculate the total time for training

        if epoch % 100 == 0:
            output_str = f"Epoch {epoch}, Likelihood {likelihood}, Epoch Time: {epoch_time / 60:.2f} mins, Total Time: {total_time / (60 * 60):.2f} hrs\n"
            print(output_str)
            with open(os.path.join(output_dir, 'log.txt'), 'a') as file:
                file.write(output_str)
            epoch_time = 0  # Reset the epoch time

        if epoch > 0 and epoch % 100 == 0:
            plot_outputs(model, data, output_dir)
            # Save the model object using pickle
            with open(os.path.join(output_dir, 'model.pkl'), 'wb') as model_file:
                pickle.dump(model, model_file)

    plot_outputs(model, data, output_dir)
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)

    metrics_results = {
        "likelihoods": likelihoods,
        "alpha_loss_increase": alpha_loss_increase,
        "gamma_loss_increase": gamma_loss_increase,
        "d1_loss_increase": d1_loss_increase,
        "d2_loss_increase": d2_loss_increase,
        "zeta_loss_increase": zeta_loss_increase,
        "chi_loss_increase": chi_loss_increase,
        "alpha_learning_rate": alpha_learning_rate,
        "gamma_learning_rate": gamma_learning_rate,
        "d1_learning_rate": d1_learning_rate,
        "d2_learning_rate": d2_learning_rate,
        "zeta_learning_rate": zeta_learning_rate,
        "chi_learning_rate": chi_learning_rate,
        "alpha_iters": alpha_iters,
        "gamma_iters": gamma_iters,
        "d1_iters": d1_iters,
        "d2_iters": d2_iters,
        "zeta_iters": zeta_iters,
        "chi_iters": chi_iters
    }

    training_results = {
        "model": model,
        "data": data,
        "true_likelihood": true_likelihood,
        "beta_s1_penalty": beta_s1_penalty,
        "beta_s2_penalty": beta_s1_penalty
    }
    return training_results, metrics_results, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Python script from the command line.')

    parser.add_argument('--tau_psi', type=int, default=1, help='Value for tau_psi')
    parser.add_argument('--tau_beta', type=int, default=1000, help='Value for tau_beta')
    parser.add_argument('--tau_s', type=int, default=1000, help='Value for tau_s')
    parser.add_argument('--num_epochs', type=int, default=2001, help='Number of training epochs')
    parser.add_argument('--beta_first', type=int, default=0, help='Whether to update beta first or G first')
    parser.add_argument('--K', type=int, default=100, help='Number of neurons')
    parser.add_argument('--R', type=int, default=15, help='Number of trials')
    parser.add_argument('--L', type=int, default=3, help='Number of latent factors')
    parser.add_argument('--intensity_mltply', type=float, default=25, help='Latent factor intensity multiplier')
    parser.add_argument('--intensity_bias', type=float, default=1, help='Latent factor intensity bias')

    args = parser.parse_args()
    K = args.K
    R = args.R
    L = args.L
    intensity_mltply = args.intensity_mltply
    intensity_bias = args.intensity_bias
    tau_psi = args.tau_psi
    tau_beta = args.tau_beta
    tau_s = args.tau_s
    num_epochs = args.num_epochs
    beta_first = args.beta_first
    notes = 'empty'

    training_results, metrics_results, output_dir = main(K, R, L, intensity_mltply, intensity_bias, tau_psi,
                                                         tau_beta, tau_s, beta_first, notes, num_epochs)
