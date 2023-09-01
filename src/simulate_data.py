import numpy as np


def generate_spike_trains(latent_factors, coeff, bias, ratio, num_trials):

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
        binned[last_binned_index:(last_binned_index+neuron_count), :] = (
            np.random.poisson(intensity[last_binned_index:(last_binned_index+neuron_count), :]))
        last_binned_index += neuron_count

    # pick only the first last_binned_index rows of binned
    binned = binned[:last_binned_index, :]
    intensity = intensity[:last_binned_index, :]
    spikes = np.where(binned >= 1)

    return intensity, binned, spikes

def generate_latent_factors(time, intensity_type=('constant', '1peak', '2peaks')):

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
