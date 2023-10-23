import numpy as np
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel

def main_function():
    self = DataAnalyzer().initialize(K=100, R=3, max_offset=0)
    binned, stim_time = self.sample_data()

    Y = binned  # K x T
    degree = 3
    L = self.latent_factors.shape[0] - 1
    model = SpikeTrainModel(Y, stim_time).initialize_for_time_warping(L, degree)

    # Training hyperparameters
    tau_psi = 1
    tau_beta = 1

    dgamma, dalpha, dzeta, dchi, dd = model.compute_analytical_grad_time_warping(tau_psi, tau_beta)
    dgamma_num, dalpha_num, dzeta_num, dchi_num, dd_num = model.compute_numerical_grad_time_warping_parallel(tau_psi, tau_beta)
    # account for constraints in dalpha and dzeta:
    dalpha[:, 1] = 0
    dzeta[:, 1] = 0

    dgamma_error = np.max(np.abs(dgamma - dgamma_num))
    dalpha_error = np.max(np.abs(dalpha - dalpha_num))
    dzeta_error = np.max(np.abs(dzeta - dzeta_num))
    dchi_error = np.max(np.abs(dchi - dchi_num))
    dd_error = np.max(np.abs(dd - dd_num))

    print(f"Max Errors: dgamma {dgamma_error}, dalpha {dalpha_error}, dzeta {dzeta_error}, dchi {dchi_error}, dd {dd_error}")

if __name__ == '__main__':
    main_function()
