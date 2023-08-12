import numpy as np
import src.simulate_data as sd
from src.manual_implemetation import log_prob, compute_lambda, compute_latent_factors, compute_numerical_grad
from src.generate_bsplines import generate_bsplines
import matplotlib.pyplot as plt

K, L, d, T, dt = 100, 5, 3, 200, 1
# Training hyperparameters
learning_rate = 1e-3
num_epochs = 1000
# base firing rate
time = np.arange(0, T, dt)/100

intensity, binned, spikes = sd.generate_spike_train(time, spike_type='2peaks', num_trials=K)
#sd.plot_intensity_and_spikes(time, intensity, binned, spikes)

# Manual Implementation
Y = binned  # K x T
B = generate_bsplines(time, d)  # T x T. The coefficient (beta) will be regularized

G = np.random.rand(K, L)
beta = np.random.rand(L, T)

G1 = np.copy(G)
beta1 = np.copy(beta)

G_grads = []
beta_grads = []
loss = []
eps=1e-6
for epoch in range(num_epochs):
    # Forward pass and gradient computation
    loss, dG, dbeta = log_prob(Y, B, G1, beta1, dt)

    # # verify gradient using finite difference
    # dG_num, dbeta_num = compute_numerical_grad(Y, B, G1, beta1, dt, eps)
    # dG_error = np.mean(np.square(dG - dG_num))
    # dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    # G_grads.append(dG_error)
    # beta_grads.append(dbeta_error)

    # Update parameters using gradients
    G1 -= learning_rate * dG
    beta1 -= learning_rate * dbeta
    # Store losses and gradients
    loss.append(loss.item())

plt.plot(np.arange(0, num_epochs), loss)
plt.show()
# plt.plot(np.arange(0, num_epochs), G_grads)
# plt.show()
# plt.plot(np.arange(0, num_epochs), beta_grads)
# plt.show()

lambda_manual = compute_lambda(B, G1, beta1)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
for i in range(K):
    plt.plot(time, lambda_manual[i, :])
plt.show()

latent_factors_manual = compute_latent_factors(B, beta1)
for i in range(L):
    plt.plot(time, latent_factors_manual[i, :])
plt.show()


# def horseshoe_prior(shape):
#   tau = torch.tensor(1.0)
#   lambdas = HalfCauchy(torch.zeros(shape), torch.ones(shape)).sample()
#   return Normal(torch.zeros(shape), tau * lambdas)
#
# # Log prior probability
# log_prior_G = horseshoe_prior(G.shape).log_prob(G).sum()
#
# # G = horseshoe_prior([K, L]).sample()

