import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def initialize_steady_grid(nx, ny, lower_t, upper_t, left_t, right_t):
    T = np.zeros((nx, ny))

    # Boundary conditions set-up
    T[0, :] = lower_t
    T[nx - 1, :] = upper_t

    T[:, 0] = left_t
    T[:, ny - 1] = right_t

    return T


def calculate_gauss_seidel(T, nx, ny, dx, dy, alpha):

    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)
    T_decay = np.copy(T)

    while current_error >= 1e-5:

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        # Calculate with Gauss Seidel method
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):
                a = (T_decay[i + 1, j] - 2 * T_decay[i, j] + T[i - 1, j]) / dx ** 2  # d2dx2
                b = (T_decay[i, j + 1] - 2 * T_decay[i, j] + T[i, j - 1]) / dy ** 2  # d2dy2
                T[i, j] = alpha * (a + b) + T_decay[i, j]

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        T_decay = np.copy(T)
        k += 1

    return T, convergence


def calculate_sor(T, nx, ny, dx, dy, alpha):

    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)
    T_decay = np.copy(T)
    omega = 1.5

    while current_error >= 1e-5:

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        # Calculate with SOR method
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):
                a = (T_decay[i + 1, j] - 2 * T_decay[i, j] + T[i - 1, j]) / dx ** 2  # d2dx2
                b = (T_decay[i, j + 1] - 2 * T_decay[i, j] + T[i, j - 1]) / dy ** 2  # d2dy2
                T[i, j] = (1 - omega) * T_decay[i, j] + omega * (alpha * (a + b) + T[i, j])

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        T_decay = np.copy(T)
        k += 1

    return T, convergence


def plot_contour(X, Y, T, solver, size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(X, Y, T[:, :], levels=20, cmap='gist_rainbow_r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(cs, ticks=[i for i in np.arange(0.0, 1.05, 0.05)], label="Phi")
    plt.savefig("homework1-task2-plots/contour-{}-{}.png".format(solver, size), format="png")
    plt.clf()


def run_main_logic(solver, size):
    # Physical parameters
    alpha = 1.172e-5  # thermal diffusivity of steel with 1% carbon
    # k = 9.7e-5 # thermal diffusivity of aluminium
    L = 1.0  # length
    W = 1.0  # width

    # Numerical parameters
    nx = size  # number of points in x direction
    ny = size  # number of points in y direction
    dk = 0.01  # time step
    max_k = 10000  # final time

    # Boundary conditions (Dirichlet)
    lower_t = 0
    upper_t = 0
    left_t = [1 - i for i in np.arange(0, 1 + 1 / (nx + 1), 1 / (nx - 1))]
    right_t = 0

    # Compute distance between nodes
    dx = L / nx
    dy = W / ny

    # Courant-Friedrichs-Lewy (CFL) condition to ensure stability
    cfl_x = alpha * dk / (dx ** 2)
    cfl_y = alpha * dk / (dy ** 2)

    if cfl_x > 0.5 or cfl_y > 0.5:
        raise TypeError('Unstable Solution!')

    # Generate 2D mesh
    X = np.linspace(0, L, nx, endpoint=True)
    Y = np.linspace(0, W, ny, endpoint=True)
    X, Y = np.meshgrid(X, Y)

    T_init = initialize_steady_grid(nx, ny, lower_t, upper_t, left_t, right_t)
    print(pd.DataFrame(T_init))
    if solver == "gauss-seidel":
        T_solved, convergence = calculate_gauss_seidel(T_init, nx, ny, dx, dy, alpha)
    elif solver == "sor":
        T_solved, convergence = calculate_sor(T_init, nx, ny, dx, dy, alpha)
    else:
        raise TypeError('Solver not valid!')

    plot_contour(X, Y, T_solved, solver, size)


def config_search_space():

    # Initialize variables
    x_values = [20, 50]
    solver_results = {}

    # Iterate over variable combinations
    for solver in ["gauss-seidel", "sor"]:
        solver_results[solver] = {}
        for size in x_values:
            print("----------------------")
            print("Current solver is {} with grid size of {}". format(solver, size))
            print("----------------------")

            run_main_logic(solver, size)

            print(" ")


if __name__ == '__main__':
    config_search_space()




