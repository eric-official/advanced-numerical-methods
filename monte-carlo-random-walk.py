import math

import matplotlib.pyplot as plt
import numpy as np


def random_walk(num_steps):
    # Generate random steps: +1 or -1
    steps = np.random.choice([-1, 1], size=num_steps)

    # Calculate cumulative sum to get the position at each step
    position = np.cumsum(steps)

    return position


def monte_carlo_random_walk(num_walks, num_steps):
    random_walks = []

    for _ in range(num_walks):
        walk = random_walk(num_steps)
        random_walks.append(walk)  # Store the final position of each walk

    return random_walks


def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))


def plot_histogram(final_positions, mu, sigma, num_steps):
    x_values = np.linspace(min(final_positions), max(final_positions), num_steps)
    pdf_values = normal_pdf(x_values, mu, sigma)

    plt.hist(final_positions, bins=20, density=True, color='b')
    plt.plot(x_values, pdf_values, color='r')
    plt.title('Histogram of Final Positions in Random Walk')
    plt.xlabel('Final Position')
    plt.ylabel('Probability')
    plt.show()
    plt.clf()


def plot_random_walks(random_walks):
    pos_one_sigma = [np.sqrt(i) for i in range(len(random_walks[0]))]
    neg_one_sigma = [-np.sqrt(i) for i in range(len(random_walks[0]))]

    pos_two_sigma = [2 * np.sqrt(i) for i in range(len(random_walks[0]))]
    neg_two_sigma = [-2 * np.sqrt(i) for i in range(len(random_walks[0]))]

    for walk in random_walks:
        plt.plot(walk, color='#61AFF2', alpha=0.5)
    plt.title('Random Walks')
    plt.xlabel('Step')
    plt.ylabel('Position')

    plt.plot(pos_one_sigma, color='y')
    plt.plot(neg_one_sigma, color='y')

    plt.plot(pos_two_sigma, color='r')
    plt.plot(neg_two_sigma, color='r')

    plt.show()
    plt.clf()


def main():
    # Parameters
    num_walks = 10000  # Number of random walks to simulate
    num_steps = 1000  # Number of steps in each random walk

    # Perform Monte Carlo simulation
    random_walks = monte_carlo_random_walk(num_walks, num_steps)

    # Plot the random walks
    plot_random_walks(random_walks)

    # Plot the results
    final_positions = [walk[-1] for walk in random_walks]
    plot_histogram(final_positions, 0, np.sqrt(1000), num_steps)


if __name__ == '__main__':
    main()
