import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps


# Generation of Sawtooth function
def falling_sawtooth_wave(t, frequency, amplitude=1.0):
    """
    Generate a falling sawtooth wave between -1 and 1.

    Parameters:
    - t: Time array.
    - frequency: Frequency of the sawtooth wave.
    - amplitude: Amplitude of the sawtooth wave (default is 1.0).

    Returns:
    - Array containing the values of the falling sawtooth wave at each time point.
    """
    sawtooth_wave_values = amplitude * (- 2 * (t * frequency - np.floor(0.5 + t * frequency)))
    return sawtooth_wave_values


def main():
    L = 1  # Periodicity of the periodic function f(x)
    frequency = 1  # No of waves in time period L
    samples = 1000
    terms = 50

    # Create a time array from 0 to 1 seconds with a given sampling rate (e.g., 1000 samples per second).
    t = np.arange(0, 1, 1 / samples)

    # Set the frequency of the sawtooth wave (e.g., 5 Hz).
    x = np.linspace(0, 1, samples, endpoint=False)
    y = falling_sawtooth_wave(t, frequency)
    
    # Calculation of Co-efficients
    a0 = 2. / L * simps(y, x)
    an = lambda n: 2.0 / L * simps(y * np.cos(2. * np.pi * n * x / L), x)
    bn = lambda n: 2.0 / L * simps(y * np.sin(2. * np.pi * n * x / L), x)

    # Sum of the series
    s = a0 / 2. + sum([an(k) * np.cos(2. * np.pi * k * x / L) + bn(k) * np.sin(2. * np.pi *
                                                                               k * x / L) for k in range(1, terms + 1)])
    # Plotting
    plt.plot(x, s, label="Fourier series")
    plt.plot(x, y, label="Original sawtooth wave")
    plt.xlabel("$x$")
    plt.ylabel("$y=f(x)$")
    plt.legend(loc='best', prop={'size': 10})
    plt.title("Sawtooth wave signal analysis by Fouries series")
    plt.savefig("fs_sawtooth.png")
    plt.show()


if __name__ == '__main__':
    main()
