import numpy as np
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel
from src.psplines_gradient_method.general_functions import compute_lambda, plot_binned, plot_spikes
import matplotlib.pyplot as plt
import time

self = DataAnalyzer().initialize()
binned, stim_time = self.sample_data()
binned_spikes = np.where(binned >= 1)
# plot_spikes(binned_spikes)
intensity, latent_factors = self.intensity, self.latent_factors
# plot_binned(binned)
# self.plot_intensity_and_latents()

Y = binned  # K x T
degree = 3
L = self.latent_factors.shape[0] - 1
model = SpikeTrainModel(Y, stim_time).initialize(L, degree)

# Training parameters
num_epochs = 5000

# Training hyperparameters
tau_beta = 100
tau_G = 2

losses = []

beta_loss_increase = []
G_loss_increase = []
d_loss_increase = []

beta_learning_rate = []
G_learning_rate = []
d_learning_rate = []

beta_iters = []
G_iters = []
d_iters = []

total_time = 0
epoch_time = 0
for epoch in range(num_epochs):
    start_time = time.time()  # Record the start time of the epoch

    result = model.log_obj_with_backtracking_line_search(tau_beta, tau_G)
    loss = result["loss"]
    beta_penalty = result["beta_penalty"]
    G_penalty = result["G_penalty"]
    dbeta = result["dlogL_dbeta"]
    dG_star = result["dlogL_dG"]
    dd = result["dlogL_dd"]

    losses.append(loss)

    beta_loss_increase.append(result["beta_loss_increase"])
    G_loss_increase.append(result["G_loss_increase"])
    d_loss_increase.append(result["d_loss_increase"])

    beta_learning_rate.append(result["smooth_beta"])
    G_learning_rate.append(result["smooth_G"])
    d_learning_rate.append(result["smooth_d"])

    beta_iters.append(result["iters_beta"])
    G_iters.append(result["iters_G"])
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
beta_learning_rate = np.array(beta_learning_rate)
G_learning_rate = np.array(G_learning_rate)
d_learning_rate = np.array(d_learning_rate)
beta_iters = np.array(beta_iters)
G_iters = np.array(G_iters)
d_iters = np.array(d_iters)
plt.plot(np.arange(0, num_epochs), losses[0:])
plt.title('Losses')
plt.show()
plt.plot(np.arange(0, num_epochs), beta_learning_rate)
plt.title('Beta Learning Rates')
plt.show()
plt.plot(np.arange(0, num_epochs), G_learning_rate)
plt.title('G Learning Rates')
plt.show()
plt.plot(np.arange(0, num_epochs), d_learning_rate)
plt.title('d Learning Rates')
plt.show()
plt.plot(np.arange(0, num_epochs), beta_iters)
plt.title('Beta Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), G_iters)
plt.title('G Iters')
plt.show()
plt.plot(np.arange(0, num_epochs), d_iters)
plt.title('d Iters')
plt.show()

lambda_manual = compute_lambda(model.V, model.d, model.G, model.beta)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(stim_time, avg_lambda_manual)
plt.show()
np.mean(np.square(intensity - lambda_manual))
for i in range(model.Y.shape[0]):
    plt.plot(stim_time, lambda_manual[i, :] + i * 2)
plt.show()

latent_factors_manual = model.beta @ model.V
for i in range(L):
    # plt.plot(np.concatenate([[stim_time[0] - 0.02, stim_time[0] - 0.01], stim_time]), model.beta[i, :])
    plt.plot(stim_time, latent_factors_manual[i, :])
    plt.title(f'Factor [{i}, :]')
plt.show()

