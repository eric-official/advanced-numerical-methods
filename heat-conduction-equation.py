import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

np.seterr(divide = 'ignore')


def initialize_grid(nx, ny, lower_t, upper_t, left_t, right_t):
    T = np.zeros((nx, ny))

    # Boundary conditions set-up
    T[:, 0] = left_t
    T[:, ny - 1] = right_t

    T[0, :] = lower_t
    T[nx - 1, :] = upper_t

    return T


def calculate_jacobi(T, nx, ny, dx, dy, alpha):

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

        # Calculate with Jacobi method
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):
                T[i, j] = 0.25 * (T_decay[i - 1, j] + T_decay[i + 1, j] + T_decay[i, j - 1] + T_decay[i, j + 1])

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        T_decay = np.copy(T)
        k += 1

    return T, convergence


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
                T[i, j] = 0.25 * (T[i - 1, j] + T_decay[i + 1, j] + T[i, j - 1] + T_decay[i, j + 1])

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
                T[i, j] = (1 - omega) * T_decay[i, j] + omega * 0.25 * (T[i - 1, j] + T_decay[i + 1, j] + T[i, j - 1] + T_decay[i, j + 1])

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        T_decay = np.copy(T)
        k += 1

    return T, convergence


def calculate_exact(T, nx, ny, dx, dy):
    equation_steps = 10

    for i in range(1, (nx - 1)):
        x = i * dx
        for j in range(1, (ny - 1)):
            y = j * dy

            sum_loop = 0
            for n in range(1, equation_steps):

                factor1 = ((-1)**(n+1)+1)/n
                factor2 = math.sin(n * math.pi * x)
                factor3 = (math.sinh(n * math.pi * y)) / (math.sinh(n * math.pi))
                sum_loop += factor1 * factor2 * factor3

            T[j, i] = (2 / math.pi) * sum_loop

    return T


def calculate_mean_absolute_error(T_solved, T_exact, nx, ny):
    abs_diff_list = []
    for i in range(nx):
        for j in range(ny):
            abs_diff = abs(T_exact[i, j] - T_solved[i, j])
            print(T_exact[i, j], T_solved[i ,j], abs_diff)
            abs_diff_list.append(abs_diff)

    mae = sum(abs_diff_list) / len(abs_diff_list)
    return mae


def plot_contour(X, Y, T, solver, size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(X, Y, T[:, :], levels=20, cmap='viridis_r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(cs, ticks=[i for i in np.arange(0.0, 1.05, 0.05)], label="Phi")
    plt.savefig("homework1-task1-plots/contour-{}-{}.png".format(solver, size), format="png")
    plt.clf()


def plot_centerline(solver_results, x_values):
    for key, value in solver_results.items():
        for el in x_values:
            plt.plot(np.linspace(0, 1, el, endpoint=True), value[el]["centerline"], label="Size "+str(el))
        plt.title("Centerline plot for {}".format(key))
        plt.xlabel("Y")
        plt.ylabel("Phi")
        plt.grid()
        plt.legend()
        plt.savefig("homework1-task1-plots/centerline-{}.png".format(key), format="png")
        plt.clf()


def plot_error_convergence(solver_results, x_values):
    for el in x_values:
        jacobi_values = solver_results["jacobi"][el]["convergence"]
        gs_values = solver_results["gauss-seidel"][el]["convergence"]
        sor_values = solver_results["sor"][el]["convergence"]

        plt.loglog(jacobi_values, label="Jacobi: {}".format(len(jacobi_values)))
        plt.loglog(gs_values, label="Gauss Seidel: {}".format(len(gs_values)), linestyle="dashed")
        plt.loglog(sor_values, label="SOR: {}".format(len(sor_values)))

        plt.xlabel('Number of Iterations')
        plt.ylabel('Residual')
        plt.grid()
        plt.title('Convergence History with grid size {}'.format(el))
        plt.legend()
        plt.savefig("homework1-task1-plots/error-convergence-{}.png".format(el), format="png")
        plt.clf()


def plot_mae(solver_results, x_values):
    px_values = [1 / i for i in x_values]

    # Create multiple lines for plot
    plt.plot(px_values, [solver_results["jacobi"][i]["mae"] for i in x_values], label="Jacobi")
    plt.plot(px_values, [solver_results["gauss-seidel"][i]["mae"] for i in x_values], label="Gauss Seidel")
    plt.plot(px_values, [solver_results["sor"][i]["mae"] for i in x_values], label="Successive over-relaxation")

    # Edit plot settings
    plt.xlabel("Delta X")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.title("Error of matrix solvers with different grid densities")
    plt.savefig("homework1-task1-plots/grid-density-errors.png", format="png")
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
    upper_t = 1
    left_t = 0
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

    T_init = initialize_grid(nx, ny, lower_t, upper_t, left_t, right_t)

    if solver == "jacobi":
        T_solved, convergence = calculate_jacobi(T_init, nx, ny, dx, dy, alpha)
    elif solver == "gauss-seidel":
        T_solved, convergence = calculate_gauss_seidel(T_init, nx, ny, dx, dy, alpha)
    elif solver == "sor":
        T_solved, convergence = calculate_sor(T_init, nx, ny, dx, dy, alpha)
    else:
        raise TypeError('Solver not valid!')

    plot_contour(X, Y, T_solved, solver, size)

    T_exact = calculate_exact(T_init, nx, ny, dx, dy)
    plot_contour(X, Y, T_exact, "Exact", size)

    centerline = T_solved[:, int(nx / 2)]
    mae = calculate_mean_absolute_error(T_solved, T_exact, nx, ny)
    return centerline, convergence, mae


def config_search_space():

    # Initialize variables
    x_values = [20, 50]
    solver_results = {}

    # Iterate over variable combinations
    for solver in ["jacobi", "gauss-seidel", "sor"]:
        solver_results[solver] = {}
        for size in x_values:
            print("----------------------")
            print("Current solver is {} with grid size of {}". format(solver, size))
            print("----------------------")

            solver_results[solver][size] = {}
            centerline, convergence, mae = run_main_logic(solver, size)
            solver_results[solver][size]["centerline"] = centerline
            solver_results[solver][size]["convergence"] = convergence
            solver_results[solver][size]["mae"] = mae

            print(" ")

    plot_centerline(solver_results, x_values)
    plot_error_convergence(solver_results, x_values)
    plot_mae(solver_results, x_values)


def manual_run(solver, size):
    if solver is None:
        raise TypeError('Solver is not configured!')
    elif size is None:
        raise TypeError('Size is not configured!')
    else:
        run_main_logic(solver, size)


if __name__ == '__main__':
    config_search_space()

