import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy import fftpack


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
    samples = 256
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
    h = a0 / 2. + sum([an(k) * np.cos(2. * np.pi * k * x / L) + bn(k) * np.sin(2. * np.pi *
                                                                               k * x / L) for k in range(1, terms + 1)])
    x4 = np.linspace(0, 4, 4 * samples, endpoint=False)
    y4 = np.tile(y, 4)
    h4 = np.tile(h, 4)

    # Plot sawtooth wave
    plt.plot(x4, h4)
    plt.xlabel("$t$")
    plt.ylabel("$y=h(t)$")
    plt.title("Sawtooth wave signal analysis with Fouries series")
    plt.grid()
    plt.ylim([-1.5, 1.5])
    plt.savefig("sawtooth.png")
    plt.show()
    plt.clf()

    # Plot fourier series expansion
    plt.plot(x4, y4)
    plt.xlabel("$t$")
    plt.ylabel("$y=h(t)$")
    plt.title("Sawtooth wave signal analysis with Fouries series")
    plt.grid()
    plt.ylim([-1.5, 1.5])
    plt.savefig("sawtooth.png")
    plt.show()
    plt.clf()

    # Do fast fourier transform
    fft = np.fft.fft(y)

    # Calculate Fast Fourier Transform
    amplitude = np.abs(fft)/samples
    power = amplitude ** 2
    angle = np.angle(fft)
    sample_freq = fftpack.fftfreq(fft.size, d=1/samples)
    amp_freq = np.array([sample_freq, amplitude])
    amp_position = amp_freq[0, :].argmax()
    peak_freq = amp_freq[1, amp_position]

    # Plot real part of FFT
    plt.plot(sample_freq[:int(samples/2)], np.imag(fft[:int(samples/2)]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Real part of FFT')
    plt.grid()
    plt.show()
    plt.clf()

    # Plot Power Spectrum
    plt.plot(sample_freq, power)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.title('Power spectrum')
    plt.grid()
    plt.show()
    plt.clf()

    # Plot Frequency
    plt.plot(sample_freq)
    plt.ylabel("Frequency [Hz]")
    plt.grid()
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
