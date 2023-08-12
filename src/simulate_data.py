import numpy as np
import matplotlib.pyplot as plt

def generate_spike_train(time, spike_type='2peaks', num_trials=100):
    dt = 0.01

    if spike_type == 'constant':
        # probability of firing is 0.2
        y = np.zeros_like(time) + 0.2 / dt
        intensity = y

    elif spike_type == 'sine':
        # Generate y values for sine function with period 2
        y = np.sin(2 * np.pi * time / 2)
        # normalize and rescale y
        intensity = 0.45 * (y - np.min(y)) / (np.max(y) - np.min(y)) + 0.05

    elif spike_type == '1peak':
        y = np.zeros_like(time)
        y[time >= 0.25] = np.sin(2 * np.pi * (time[time >= 0.25] - 0.25) / 0.75)
        y[time > 0.625] = 0
        intensity = 0.45 * (y - np.min(y)) / (np.max(y) - np.min(y)) + 0.05

    elif spike_type == '2peaks':
        y = np.zeros_like(time)
        y[time >= 0.25] = np.sin(2 * np.pi * (time[time >= 0.25] - 0.25) / 0.75)
        y[time > 0.625] = 0
        y[time >= 1.25] = np.sin(2 * np.pi * (time[time >= 1.25] - 1.25) / 0.75)
        y[time > 1.625] = 0
        intensity = 0.4 * (y - np.min(y)) / (np.max(y) - np.min(y)) + 0.1

    binned = np.random.binomial(1, intensity, size=(num_trials, len(intensity)))
    spikes = np.where(binned == 1)

    return intensity, binned, spikes

def plot_intensity_and_spikes(t, r, b, s):
    # t is time
    # r is intensity/rate
    # b is binned spikes
    # s is spikes
    plt.plot(t, r)
    plt.show()

    _, ax = plt.subplots()
    ax.imshow(b)
    ax.invert_yaxis()
    plt.show()

    # Group entries by unique values of s[0]
    unique_s_0 = np.unique(s[0])
    grouped_s = []
    for i in unique_s_0:
        indices = np.where(s[0] == i)[0]
        values = s[1][indices]
        grouped_s.append((i, values))
    for group in grouped_s:
        plt.scatter(group[1], np.zeros_like(group[1]) + group[0], s=1, c='black')
    plt.show()
