import numpy as np
import src.simulate_data as sd
from src.psplines_gradient_method.manual_implemetation import log_prob, log_obj
from src.psplines_gradient_method.general_functions import compute_lambda, compute_latent_factors, \
    compute_numerical_grad, create_first_diff_matrix, create_second_diff_matrix, plot_intensity_and_latents, \
    plot_binned, plot_spikes
from src.psplines_gradient_method.generate_bsplines import generate_bsplines
import matplotlib.pyplot as plt

K, degree, T = 100, 3, 200
intensity_type = ('constant', '1peak', '2peaks')
L = len(intensity_type) - 1
# base firing rate
time = np.arange(0, T, 1)/100
dt = time[1] - time[0]

latent_factors = sd.generate_latent_factors(time, intensity_type=intensity_type)
np.random.seed(0)
intensity, binned, spikes = sd.generate_spike_trains(latent_factors, (0.1, 0.13, 0.13), (-3, -3, -3), (1/3, 1/3, 1/3), K)
K = binned.shape[0]

# Manual Implementation
Y = binned  # K x T
B = generate_bsplines(time, degree)  # T x T. The coefficient (beta) will be regularized
P = B.shape[0]

np.random.seed(0)
G = np.random.rand(K, L)
np.random.seed(0)
beta = np.random.rand(L, P)
np.random.seed(0)
d = np.random.rand(K)

# Training hyperparameters
tau_beta = 100
tau_G = 2
tau_d = 3
smooth_beta = 800
smooth_G = 800
smooth_d = 400
Omega = create_first_diff_matrix(P)

G_grads = []
beta_grads = []
d_grads = []
G_smooths = []
beta_smooths = []
d_smooths = []
losses = []
eps = 1e-4
for epoch in range(10):
    # Forward pass and gradient computation

    result = log_obj(Y, B, d, G, beta, Omega, tau_beta, tau_G, tau_d, smooth_beta, smooth_G, smooth_d, dt)
    loss = result["loss"]
    dd = result["dLogL_dd"]
    dG = result["dlogL_dG"]
    dbeta = result["dlogL_dbeta"]
    log_likelihood = result["log_likelihood"]
    beta_penalty = result["beta_penalty"]
    G_penalty = result["G_penalty"]
    d_penalty = result["d_penalty"]

    # verify gradient using finite difference
    dd_num, dG_num, dbeta_num = compute_numerical_grad(Y, B, d, G, beta, Omega,
                                                       tau_beta, tau_G, tau_d,
                                                       smooth_beta, smooth_G,
                                                       smooth_d, dt, log_obj, eps)
    dd_error = np.mean(np.square(dd - dd_num))
    dG_error = np.mean(np.square(dG - dG_num))
    dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    d_grads.append(dd_error)
    G_grads.append(dG_error)
    beta_grads.append(dbeta_error)
    print(f"Epoch {epoch}, dd_error {dd_error}, dG_error {dG_error}, dbeta_error {dbeta_error}")

    # Update parameters using gradients
    d = result["d_plus"]
    G = result["G_plus"]
    beta = result["beta_plus"]

