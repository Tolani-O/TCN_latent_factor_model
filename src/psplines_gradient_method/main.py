import numpy as np
import src.simulate_data as sd
from src.psplines_gradient_method.manual_implemetation import log_prob, compute_lambda, compute_latent_factors, \
    compute_numerical_grad, log_obj
from src.psplines_gradient_method.generate_bsplines import generate_bsplines
import matplotlib.pyplot as plt

K, degree, T = 100, 3, 200
intensity_type = ('constant', '1peak', '2peaks')
L = len(intensity_type)
# Training hyperparameters
learning_rate = 1e-3
num_epochs = 50000
# base firing rate
time = np.arange(0, T, 1)/100
dt = time[1] - time[0]

latent_factors = sd.generate_latent_factors(time, intensity_type=intensity_type)
np.random.seed(0)
intensity, binned, spikes = sd.generate_spike_trains(latent_factors, (0.1, 0.13, 0.13), (-3, -3, -3), (1/3, 1/3, 1/3), K)
K = binned.shape[0]

# sd.plot_intensity_and_spikes(time, latent_factors, intensity, binned, spikes)

# Manual Implementation
Y = binned  # K x T
B = generate_bsplines(time, degree)  # T x T. The coefficient (beta) will be regularized
# start = 190
# num_basis = 10
# for i in range(num_basis):
#     plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
# plt.show()

np.random.seed(0)
G = np.random.rand(K, L)
np.random.seed(0)
beta = np.random.rand(L, B.shape[0])
np.random.seed(0)
d = np.random.rand(K)
tausq = 100*np.ones(L) # 10*np.square(np.random.rand(L))

G_grads = []
beta_grads = []
d_grads = []
tausq_grads = []
losses = []
eps=1e-4
for epoch in range(num_epochs):
    # Forward pass and gradient computation

    # tausq = 2*tausq
    # loss, dd, dG, dbeta, dtausq = log_prob(Y, B, d, G, beta, tausq, dt)
    #
    # # # verify gradient using finite difference
    # # dd_num, dG_num, dbeta_num, dtausq_num = compute_numerical_grad(Y, B, d, G, beta, tausq, dt, log_prob, eps)
    # # dd_error = np.mean(np.square(dd - dd_num))
    # # dG_error = np.mean(np.square(dG - dG_num))
    # # dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    # # dtausq_error = np.mean(np.square(dtausq - dtausq_num))
    # # d_grads.append(dd_error)
    # # G_grads.append(dG_error)
    # # beta_grads.append(dbeta_error)
    # # tausq_grads.append(dtausq_error)
    # # print(f"Epoch {epoch}, dd_error {dd_error}, dG_error {dG_error}, dbeta_error {dbeta_error}, dtausq_error {dtausq_error}")

    result = log_obj(Y, B, d, G, beta, tausq, dt)
    loss, dd, dG, dbeta, lk, lp = (
        result["loss"], result["dLogL_dd"], result["dlogL_dG"], result["dlogL_dbeta"], result["log_likelihood"], result["penalty"])
    # epoch=0
    # print(f"Epoch {epoch}, loss {lk}, penalty {lp}")

    # # verify gradient using finite difference
    # dd_num, dG_num, dbeta_num, _ = compute_numerical_grad(Y, B, d, G, beta, tausq, dt, log_obj, eps)
    # dd_error = np.mean(np.square(dd - dd_num))
    # dG_error = np.mean(np.square(dG - dG_num))
    # dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    # d_grads.append(dd_error)
    # G_grads.append(dG_error)
    # beta_grads.append(dbeta_error)
    # print(f"Epoch {epoch}, dd_error {dd_error}, dG_error {dG_error}, dbeta_error {dbeta_error}")

    # Update parameters using gradients
    d -= learning_rate * dd
    G -= learning_rate * dG
    beta -= learning_rate * dbeta
    # tausq -= learning_rate * dtausq
    # Store losses and gradients
    losses.append(loss)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

plt.plot(np.arange(0, num_epochs), losses)
plt.show()

lambda_manual = compute_lambda(B, d, G, beta)
avg_lambda_manual = np.mean(lambda_manual, axis=0)
plt.plot(time, avg_lambda_manual)
plt.show()
np.mean(np.square(intensity - lambda_manual))
for i in range(K):
    plt.plot(time, lambda_manual[i, :] + i*10)
plt.show()

latent_factors_manual = compute_latent_factors(B, beta)
for i in range(L):
    plt.plot(np.concatenate([[time[0]-dt], time]), beta[i, :])
    plt.plot(time, latent_factors_manual[i, :])
    plt.title(f'beta[{i}, :]')
    plt.show()

# for i in range(L):
#     plt.plot(time, latent_factors_manual[i, :])
# plt.show()


# def horseshoe_prior(shape):
#   tau = torch.tensor(1.0)
#   lambdas = HalfCauchy(torch.zeros(shape), torch.ones(shape)).sample()
#   return Normal(torch.zeros(shape), tau * lambdas)
#
# # Log prior probability
# log_prior_G = horseshoe_prior(G.shape).log_prob(G).sum()
#
# # G = horseshoe_prior([K, L]).sample()

