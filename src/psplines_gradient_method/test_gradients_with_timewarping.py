import numpy as np
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel


self = DataAnalyzer().initialize()
binned, time = self.sample_data()

Y = binned  # K x T
degree = 3
L = self.latent_factors.shape[0] - 1

# Training hyperparameters
tau_psi = 80
tau_beta = 80
tau_G = 2

model = SpikeTrainModel(Y, time).initialize_for_time_warping(L, degree)

epoch = 0
for epoch in range(10):

    dpsi, dbeta, dG, dd = model.compute_analytical_grad_time_warping(tau_beta)
    dpsi_num, dbeta_num, dG_num, dd_num = model.compute_numerical_grad_time_warping(tau_beta, tau_G)

    dpsi_error = np.mean(np.square(dpsi - dpsi_num))
    dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    dG_error = np.mean(np.square(dG - dG_num))
    dd_error = np.mean(np.square(dd - dd_num))

    self.log_obj_with_backtracking_line_search_and_time_warping(tau_psi, tau_beta, tau_G)

    print(f"Epoch {epoch}, dpsi_error {dpsi_error}, dbeta_error {dbeta_error}, dG_error {dG_error}, dd_error {dd_error}")

# ideas for debugging:
# 1. use B as B_psi and compute the gradients. They should be the same as the gradients computed without time warping.
#       validate that the gradients of d, beta and G this way.
# 2. Pass in the gradients into the function, compute the difference elementwise, and see where the difference is coming from
