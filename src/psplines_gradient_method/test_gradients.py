import numpy as np
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel


self = DataAnalyzer().initialize()
binned, time = self.sample_data()

Y = binned  # K x T
degree = 3
L = self.latent_factors.shape[0] - 1

# Training hyperparameters
tau_beta = 80
tau_G = 2

model = SpikeTrainModel(Y, time).initialize(L, degree)

epoch = 0
for epoch in range(10):

    dbeta, dG, dd = model.compute_analytical_grad(tau_beta)
    dbeta_num, dG_num, dd_num = model.compute_numerical_grad(tau_beta, tau_G)

    dbeta_error = np.mean(np.square(dbeta - dbeta_num))
    dG_error = np.mean(np.square(dG - dG_num))
    dd_error = np.mean(np.square(dd - dd_num))

    self.log_obj_with_backtracking_line_search(tau_beta, tau_G)

    print(f"Epoch {epoch}, dbeta_error {dbeta_error}, dG_error {dG_error}, dd_error {dd_error}")
