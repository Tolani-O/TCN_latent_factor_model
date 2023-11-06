import numpy as np
from src.simulate_data import DataAnalyzer
from src.psplines_gradient_method.SpikeTrainModel import SpikeTrainModel

def main_function():
    self = DataAnalyzer().initialize(K=100, R=1, max_offset=0)
    binned, stim_time = self.sample_data()

    Y = binned  # K x T
    degree = 3
    L = self.latent_factors.shape[0] - 1
    model = SpikeTrainModel(Y, stim_time).initialize_for_time_warping(L, degree)

    # Training hyperparameters
    tau_psi = 1
    tau_beta = 1
    tau_s = 1

    dgamma, dd1, dd2, dalpha, dzeta, dchi, dc = model.compute_analytical_grad_time_warping(tau_psi, tau_beta, tau_s)
    dgamma_num, dd1_num, dd2_num, dalpha_num, dzeta_num, dchi_num, dc_num = model.compute_numerical_grad_time_warping_parallel(tau_psi, tau_beta, tau_s)
    # account for constraints in dalpha and dzeta:
    dalpha[:, 1] = 0 # might need to fix dalpha[:, 0] = 0 as well
    dzeta[:, 1] = 0 # might need to fix dzeta[:, 0] = 0 as well
    dchi[:, 0] = 0
    dd1[:, 0] = 0
    dd2[:, 0] = 0


    dgamma_error = np.max(np.abs(dgamma - dgamma_num))
    dd1_error = np.max(np.abs(dd1 - dd1_num))
    dd2_error = np.max(np.abs(dd2 - dd2_num))
    dalpha_error = np.max(np.abs(dalpha - dalpha_num))
    dzeta_error = np.max(np.abs(dzeta - dzeta_num))
    dchi_error = np.max(np.abs(dchi - dchi_num))
    dc_error = np.max(np.abs(dc - dc_num))

    print(f"Max Errors: dgamma {dgamma_error}, dd1 {dd1_error}, dd2 {dd2_error}, dalpha {dalpha_error}, dzeta {dzeta_error}, dchi {dchi_error}, dc {dc_error}")

if __name__ == '__main__':
    main_function()
