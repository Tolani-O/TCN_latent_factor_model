import os
import numpy as np
import matplotlib.pyplot as plt


class DataAnalyzer:

    def __init__(self):
        self.time = None
        self.latent_factors = None
        self.intensity = None
        self.binned = None


    def initialize(self, K=100, T=200, seed=0, intensity_type=('constant', '1peak', '2peaks'), max_offset=0):
        time = np.arange(0, T, 1) / 100
        np.random.seed(seed)
        self.latent_factors = self.generate_latent_factors(time, intensity_type)
        np.random.seed(seed)
        # might need to move coefficients and stuff to be passed in as parameters
        self.intensity, self.binned = self.generate_spike_trains(self.latent_factors,
                                                                 (0.1, 0.13, 0.13),
                                                                 (-3, -3, -3),
                                                                 (1/3, 1/3, 1/3),
                                                                 K, max_offset)
        self.time = time
        return self

    def generate_latent_factors(self, time, intensity_type):

        # intensity_type is a string
        if isinstance(intensity_type, str):
            intensity_type = [intensity_type]

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

            latent_factors[i, :] = 15 * rate + 5

        return latent_factors


    def generate_spike_trains(self, latent_factors, coeff, bias, ratio, num_trials, max_offset):

        num_factors, num_timesteps = latent_factors.shape
        if num_factors == 1:
            ratio = [1]
        # make sure intensity_type and ratio have the same length
        assert num_factors == len(ratio)
        # make sure ratio sums to 1
        ratio = np.array(ratio)
        ratio = ratio / np.sum(ratio)

        intensity = np.zeros((num_trials, num_timesteps))
        binned = np.zeros((num_trials, num_timesteps))
        last_binned_index = 0
        # loop over the rows of latent_factors
        for i in range(num_factors):
            neuron_count = int(num_trials * ratio[i])
            intensity[last_binned_index:(last_binned_index+neuron_count), :] = np.vstack([np.exp(bias[i] + coeff[i] * latent_factors[i, :])] * neuron_count)
            # Add random integer offset to each row of intensity
            for j in range(last_binned_index, last_binned_index + neuron_count):
                offset = np.random.randint(-max_offset, max_offset + 1)  # Random offset between -max_offset and max_offset
                intensity[j, :] = np.roll(intensity[j, :], offset)
            binned[last_binned_index:(last_binned_index+neuron_count), :] = (
                np.random.poisson(intensity[last_binned_index:(last_binned_index+neuron_count), :]))
            last_binned_index += neuron_count

        # pick only the first last_binned_index rows of binned
        binned = binned[:last_binned_index, :]
        intensity = intensity[:last_binned_index, :]

        return intensity, binned


    def sample_data(self):
        return self.binned, self.time


    def plot_intensity_and_latents(self):

        time, latent_factors, intensity = self.time, self.latent_factors, self.intensity
        # plot latent factors
        plt.figure()
        for i in range(latent_factors.shape[0]):
            plt.plot(time, latent_factors[i, :] + i)
        plt.savefig(os.path.join(os.getcwd(), 'outputs', 'groundTruth_latent_factors.png'))

        # plot neuron intensities
        plt.figure()
        for i in range(intensity.shape[0]):
            plt.plot(time, intensity[i, :] + i * 0.1)
        plt.savefig(os.path.join(os.getcwd(), 'outputs', 'groundTruth_intensities.png'))
