import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy import fftpack, signal
import os


# Generation of Sawtooth function
def falling_sawtooth_wave(t, frequency, amplitude=1.0):
    sawtooth_wave_values = amplitude * (- 2 * (t * frequency - np.floor(0.5 + t * frequency)))
    return sawtooth_wave_values


def main():
    L = 1  # Periodicity of the periodic function f(x)
    frequency = 1  # No of waves in time period L
    samples = 256
    k = 50 # Number of terms in the fourier series

    # Create a time array from 0 to 1 seconds with a given sampling rate (e.g., 1000 samples per second).
    t = np.arange(0, 1, 1 / samples)

    # Set the frequency of the sawtooth wave (e.g., 5 Hz).
    x = np.linspace(0, 1, samples, endpoint=False)
    y = falling_sawtooth_wave(t, frequency)

    # Calculation of Co-efficients
    a0 = 2. / L * simps(y, x)
    an = lambda n: 2.0 / L * simps(y * np.cos(2. * np.pi * n * x / L), x)
    bn = lambda n: 2.0 / L * simps(y * np.sin(2. * np.pi * n * x / L), x)
    print("a0", a0)
    # Sum of the series
    h = a0 / 2. + sum([an(k) * np.cos(2. * np.pi * k * x / L) + bn(k) * np.sin(2. * np.pi *
                                                                               k * x / L) for k in range(1, k + 1)])

    # Repeat the sawtooth wave 4 times
    x4 = np.linspace(0, 4, 4 * samples, endpoint=False)
    y4 = np.tile(y, 4)
    h4 = np.tile(h, 4)

    # Plot sawtooth wave
    plt.plot(x4, y4)
    plt.xlabel("$t$")
    plt.ylabel("$y=h(t)$")
    plt.title("Sawtooth wave signal analysis without Fouries series")
    plt.grid()
    plt.ylim([-1.5, 1.5])
    plt.savefig("homework2-task3-plots/sawtooth-wave.png", format="png")
    plt.clf()

    # Plot fourier series expansion
    plt.plot(x4, h4)
    plt.xlabel("$t$")
    plt.ylabel("$y=h(t)$")
    plt.title("Sawtooth wave signal analysis with Fouries series")
    plt.grid()
    plt.ylim([-1.5, 1.5])
    plt.savefig("homework2-task3-plots/sawtooth-fourier.png", format="png")
    plt.clf()

    # Do fast fourier transform
    fft = np.fft.fft(y)

    # Calculate Fast Fourier Transform
    amplitude = np.abs(fft)/samples
    power = amplitude ** 2
    sample_freq = fftpack.fftfreq(fft.size, d=1/samples)

    # Plot real part of FFT
    plt.plot(sample_freq[1:int(samples / 2)], np.real(fft[1:int(samples / 2)]))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Real part of FFT")
    plt.grid()
    plt.savefig("homework2-task3-plots/fft-real.png", format="png")
    plt.clf()

    # Plot imaginary part of FFT
    plt.plot(sample_freq[:int(samples/2)], np.imag(fft[:int(samples/2)]))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Real part of FFT")
    plt.grid()
    plt.savefig("homework2-task3-plots/fft-imaginary.png", format="png")
    plt.clf()

    # Plot Power Spectrum
    plt.plot(sample_freq, power)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")
    plt.title("Power spectrum")
    plt.grid()
    plt.savefig("homework2-task3-plots/fft-power-spectrum.png", format="png")
    plt.clf()

    # Plot Frequency
    plt.plot(sample_freq)
    plt.ylabel("Frequency [Hz]")
    plt.grid()
    plt.savefig("homework2-task3-plots/fft-frequency.png", format="png")
    plt.clf()


if __name__ == "__main__":
    # if the demo_folder directory is not present then create it
    if not os.path.exists("homework2-task3-plots"):
        os.makedirs("homework2-task3-plots")

    main()
