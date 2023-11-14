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


def plot_histogram(final_positions):
    plt.hist(final_positions, bins=20, density=True, color='b')
    plt.title('Histogram of Final Positions in Random Walk')
    plt.xlabel('Final Position')
    plt.ylabel('Probability')
    plt.show()
    plt.clf()


def plot_random_walks(random_walks):
    for walk in random_walks:
        plt.plot(walk, color='b', alpha=0.3)
    plt.title('Random Walks')
    plt.xlabel('Step')
    plt.ylabel('Position')
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
    plot_histogram(final_positions)


if __name__ == '__main__':
    main()
