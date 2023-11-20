import numpy as np
from scipy.sparse import csr_array
from src.psplines_gradient_method.general_functions import create_second_diff_matrix


class DataAnalyzer:

    def __init__(self):
        self.time = None
        self.latent_factors = None
        self.intensity = None
        self.binned = None
        self.latent_coupling = None


    def initialize(self, K=100, T=200, R=3, intensity_type=('constant', '1peak', '2peaks'),
                   coeff=(1, 1, 1), ratio=(1/3, 1/3, 1/3),
                   intensity_mltply=15, intensity_bias=5, max_offset=0):
        self.time = np.arange(0, T, 1) / 100
        self.latent_factors = self.generate_latent_factors(intensity_type, intensity_mltply, intensity_bias)
        self.intensity, self.binned, self.latent_coupling = self.generate_spike_trains(coeff, ratio, K, R, max_offset)
        return self

    def generate_latent_factors(self, intensity_type, intensity_mltply, intensity_bias):

        # intensity_type is a string
        if isinstance(intensity_type, str):
            intensity_type = [intensity_type]

        time = self.time
        num_timesteps = len(time)
        latent_factors = np.zeros((len(intensity_type), num_timesteps))
        for i, itype in enumerate(intensity_type):
            rate = np.zeros_like(time)

            if itype == 'constant':
                nothing_to_do = True

            elif itype == 'sine':
                # Generate y values for sine function with period 2
                rate = np.sin(2 * np.pi * time / 2)

            elif itype == '1peak':
                rate[time >= 0.25] = np.sin(2 * np.pi * (time[time >= 0.25] - 0.25) / 0.75)
                rate[time > 0.625] = 0

            elif itype == '2peaks':
                rate[time >= 0.25] = np.sin(2 * np.pi * (time[time >= 0.25] - 0.25) / 0.75)
                rate[time > 0.625] = 0
                rate[time >= 1.25] = np.sin(2 * np.pi * (time[time >= 1.25] - 1.25) / 0.75)
                rate[time > 1.625] = 0

            latent_factors[i, :] = intensity_mltply * rate + intensity_bias

        return latent_factors


    def generate_spike_trains(self, coeff, ratio, num_neurons, num_trials, max_offset):

        latent_factors = self.latent_factors
        num_factors, num_timesteps = latent_factors.shape
        if num_factors == 1:
            ratio = [1]
        # make sure intensity_type and ratio have the same length
        assert num_factors == len(ratio)
        # make sure ratio sums to 1
        ratio = np.array(ratio)
        ratio = ratio / np.sum(ratio)

        intensity = np.zeros((num_neurons, num_trials*num_timesteps))
        latent_coupling = np.zeros((num_neurons, num_factors))
        binned = np.zeros((num_neurons, num_trials*num_timesteps))
        latent_factor_trials = np.hstack([latent_factors] * num_trials)
        dt = round(self.time[1] - self.time[0], 3)
        last_binned_index = 0
        # loop over the rows of latent_factors
        for i in range(num_factors):
            neuron_count = int(num_neurons * ratio[i])
            intensity[last_binned_index:(last_binned_index+neuron_count), :] = np.vstack([coeff[i] * latent_factor_trials[i, :]] * neuron_count)
            latent_coupling[last_binned_index:(last_binned_index+neuron_count), i] = 1
            # Add random integer offset to each row of intensity
            for j in range(last_binned_index, last_binned_index + neuron_count):
                offset = np.random.randint(-max_offset, max_offset + 1)  # Random offset between -max_offset and max_offset
                intensity[j, :] = np.roll(intensity[j, :], offset)
            binned[last_binned_index:(last_binned_index+neuron_count), :] = (np.random.poisson(intensity[last_binned_index:(last_binned_index+neuron_count), :] * dt))
            last_binned_index += neuron_count

        # pick only the first last_binned_index rows of binned
        binned = binned[:last_binned_index, :]
        intensity = intensity[:last_binned_index, :]
        latent_coupling = latent_coupling[:last_binned_index, :]

        return intensity, binned, latent_coupling


    def sample_data(self):
        return self.binned, self.time

    def likelihood(self, tau_beta):
        intensity = self.intensity
        binned = self.binned
        T = self.time.shape[0]
        Delta2 = csr_array(create_second_diff_matrix(T))
        dt = round(self.time[1] - self.time[0], 3)
        likelihood = np.sum(np.log(intensity) * binned - intensity * dt)
        beta_s2_penalty = - tau_beta * 1/self.latent_factors.shape[0] * np.sum((Delta2 @ self.latent_factors.T)**2)
        return likelihood, beta_s2_penalty
