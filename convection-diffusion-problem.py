import math
import os

import matplotlib.pyplot as plt
import numpy as np


def calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho):

    # Initialize variables
    a_w = np.zeros((nx, ny))
    a_e = np.zeros((nx, ny))
    a_n = np.zeros((nx, ny))
    a_s = np.zeros((nx, ny))

    # Iterate over grid
    for i in range(1, nx + 1):
        dx = xnode[i] - xnode[i - 1]
        for j in range(1, ny + 1):

            # Calculate finite volume with upwind scheme
            dy = ynode[j] - ynode[j - 1]
            delx_e = xcell[i + 1] - xcell[i]
            delx_w = xcell[i] - xcell[i - 1]
            dely_n = ycell[j + 1] - ycell[j]
            dely_s = ycell[j] - ycell[j - 1]
            a_w[i - 1, j - 1] = gamma * dy / delx_w - min(rho * u[i - 1, j] * dy * (-1), 0)
            a_e[i - 1, j - 1] = gamma * dy / delx_e - min(rho * u[i, j] * dy, 0)
            a_n[i - 1, j - 1] = gamma * dx / dely_n - min(rho * v[i, j] * dx, 0)
            a_s[i - 1, j - 1] = gamma * dx / dely_s - min(rho * v[i, j - 1] * dx * (-1), 0)

    return a_e, a_w, a_n, a_s


def calculate_gauss_seidel(T, nx, ny, ycell, a_e, a_w, a_n, a_s, B=None, a_p_im=None):
    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)

    # Calculate a_p
    a_p = np.zeros((nx, ny))
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            a_p[i - 1, j - 1] = a_e[i - 1, j - 1] + a_w[i - 1, j - 1] + a_n[i - 1, j - 1] + a_s[i - 1, j - 1]

    # Calculate with Gauss Seidel method until error threshold is reached
    while current_error >= 1e-5:

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)) and B is None:
            print("Current error", current_error)
        decay_error = current_error

        T_decay = T.copy()
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):

                # Set flow conditions
                if i == 1:
                    t_w = 1 - ycell[j]
                else:
                    t_w = T_decay[i - 1, j]
                if i == nx:
                    t_e = T_decay[i, j]
                else:
                    t_e = T_decay[i + 1, j]
                if j == 1:
                    t_s = T_decay[i, j]
                else:
                    t_s = T_decay[i, j - 1]
                if j == ny:
                    t_n = 0.0
                else:
                    t_n = T_decay[i, j + 1]

                # Calculate new temperature with depending on if calculation is for implicit euler or not
                T_old = T[i, j]
                if B is None:
                    T_new = (a_e[i - 1, j - 1] * t_e + a_w[i - 1, j - 1] * t_w + a_n[i - 1, j - 1] * t_n + a_s[
                        i - 1, j - 1] * t_s) / a_p[i - 1, j - 1]
                else:
                    T_new = (a_e[i - 1, j - 1] * t_e + a_w[i - 1, j - 1] * t_w + a_n[i - 1, j - 1] * t_n + a_s[
                        i - 1, j - 1] * t_s + B[(j - 2) * nx + i - 1]) / a_p_im[i - 1, j - 1]
                T[i, j] = T_old + (T_new - T_old)

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        k += 1

    return T, convergence


def calculate_sor(T, nx, ny, ycell, a_e, a_w, a_n, a_s):
    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)
    omega = 1.1

    a_p = np.zeros((nx, ny))
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            a_p[i - 1, j - 1] = a_e[i - 1, j - 1] + a_w[i - 1, j - 1] + a_n[i - 1, j - 1] + a_s[i - 1, j - 1]

    while current_error >= 1e-5:

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        T_decay = T.copy()
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):

                # Set flow conditions
                if i == 1:
                    Tw = 1 - ycell[j]
                else:
                    Tw = T_decay[i - 1, j]
                if i == nx:
                    Te = T_decay[i, j]
                else:
                    Te = T_decay[i + 1, j]
                if j == 1:
                    Ts = T_decay[i, j]
                else:
                    Ts = T_decay[i, j - 1]
                if j == ny:
                    Tn = 0.0
                else:
                    Tn = T_decay[i, j + 1]

                # Calculate new temperature
                T_old = T[i, j]
                T_new = (a_e[i - 1, j - 1] * Te + a_w[i - 1, j - 1] * Tw + a_n[i - 1, j - 1] * Tn + a_s[
                    i - 1, j - 1] * Ts) / a_p[i - 1, j - 1]
                T[i, j] = T_old + omega * (T_new - T_old)

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        k += 1

    return T, convergence


def calculate_explicit_euler(T, max_k, dk, nx, ny, ymax, xnode, ynode, ycell, a_e, a_w, a_n, a_s, rho):
    a_p = np.zeros((nx, ny))
    for i in range(1, nx + 1):
        dx = xnode[i] - xnode[i - 1]
        for j in range(1, ny + 1):
            dy = ynode[j] - ynode[j - 1]
            ap0 = rho * dx * dy / dk
            a_p[i - 1, j - 1] = ap0

    current_error = decay_error = 10
    convergence = np.array([], float)

    # Calculate with explicit Euler method for max_k steps
    for k in np.arange(0, max_k):

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        T_decay = T.copy()
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):

                # Set flow conditions
                if i == 1:
                    t_w = 1 - ycell[j] / ymax
                else:
                    t_w = T_decay[i - 1, j]
                if i == nx + 1:
                    t_e = T_decay[i, j]
                else:
                    t_e = T_decay[i + 1, j]
                if j == 1:
                    t_s = T_decay[i, j]
                else:
                    t_s = T_decay[i, j - 1]
                if j == ny + 1:
                    t_n = 0.0
                else:
                    t_n = T_decay[i, j + 1]

                # Calculate new temperature
                t_p = T_decay[i, j]
                T[i, j] = a_e[i - 1, j - 1] * t_e + a_w[i - 1, j - 1] * t_w + a_n[i - 1, j - 1] * t_n + a_s[
                    i - 1, j - 1] * t_s + (
                                  ap0 - a_e[i - 1, j - 1] - a_w[i - 1, j - 1] - a_n[i - 1, j - 1] - a_s[
                              i - 1, j - 1]) * t_p
                T[i, j] = T[i, j] / a_p[i - 1, j - 1]

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)

    return T, convergence


def calculate_implicit_euler(B, T, max_k, dk, nx, ny, ymax, rho, xnode, xcell, ynode, ycell, gamma, u, v):
    ap0 = np.zeros((nx, ny))
    a_p = np.zeros((nx, ny))

    current_error = decay_error = 10
    convergence = np.array([], float)

    # Calculate with implicit Euler method for max_k steps
    for k in np.arange(0, max_k):

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        T_decay = T.copy()
        a_e, a_w, a_n, a_s = calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho)

        for i in range(1, nx + 1):
            dx = xnode[i] - xnode[i - 1]
            for j in range(1, ny + 1):
                dy = ynode[j] - ynode[j - 1]
                ap0[i - 1, j - 1] = rho * dx * dy / dk
                a_p[i - 1, j - 1] = a_w[i - 1, j - 1] + a_e[i - 1, j - 1] + a_n[i - 1, j - 1] + a_s[i - 1, j - 1] + ap0[
                    i - 1, j - 1]
                B[(j - 2) * nx + i - 1] = ap0[i - 1, j - 1] * T_decay[i - 1, j - 1]

        # Calculate B coefficient
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                B[(j - 2) * nx + i - 1] = ap0[i - 1, j - 1] * T_decay[i, j]

        for i in range(0, nx):
            for j in range(0, ny):

                # Set flow conditions
                if i == 0:
                    Tval = 1 - ycell[j + 1] / ymax
                    B[(j - 1) * nx + i] = a_w[i, j] * Tval + B[(j - 1) * nx + i]
                    a_w[i, j] = 0.0
                elif i == nx - 1:
                    a_p[i, j] = a_p[i, j] - a_e[i, j]
                    a_e[i, j] = 0.0
                if j == 0:
                    a_p[i, j] = a_p[i, j] - a_s[i, j]
                    a_s[i, j] = 0.0
                elif j == ny - 1:
                    Tval = 0.0
                    B[(j - 1) * nx + i] = a_n[i, j] * Tval + B[(j - 1) * nx + i]
                    a_n[i, j] = 0.0

        # Calculate new temperature with Gauss Seidel method
        T = np.zeros((nx + 2, ny + 2))
        T, _ = calculate_gauss_seidel(T, nx, ny, ycell, a_e, a_w, a_n, a_s, B, a_p)

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)

    return T, convergence


def plot_contour(X, Y, T, solver, size, nx, ny, ycell, ymax, gamma=None):

    # Apply boundary conditions
    for i in range(nx + 2):
        for j in range(ny + 2):
            if i == 0:
                T[i, j] = 1 - ycell[j] / ymax
            elif i == nx + 1:
                T[i, j] = T[i - 1, j]
            elif j == 0:
                T[i, j] = T[i, j + 1]
            elif j == ny + 1:
                T[i, j] = 0.0

    # Plot contour
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(X, Y, T[:, :], levels=20, cmap='viridis_r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(cs, ticks=[i for i in np.arange(0.0, 1.05, 0.05)], label="Phi")
    if gamma is not None:
        plt.savefig("homework1-task2-plots/contour-{}-{}-g{}.png".format(solver, size, gamma), format="png")
    else:
        plt.savefig("homework1-task2-plots/contour-{}-{}.png".format(solver, size), format="png")
    plt.clf()


def plot_error_convergence(conv_results, x_value, gamma=None):
    if gamma is not None:
        gs_values = conv_results["gauss-seidel"]
        sor_values = conv_results["sor"]
        plt.loglog(gs_values, label="Gauss Seidel: {}".format(len(gs_values)), linestyle="dashed")
        plt.loglog(sor_values, label="SOR: {}".format(len(sor_values)))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Residual')
        plt.grid()
        plt.title('Convergence History of steady equation with\n grid size {} and gamma={}'.format(x_value, gamma))
        plt.legend()
        plt.savefig("homework1-task2-plots/error-convergence-steady-{}-g{}.png".format(x_value, gamma), format="png")
        plt.clf()

    else:
        explicit_values = conv_results["explicit"]
        implicit_values = conv_results["implicit"]
        plt.loglog(explicit_values, label="Explicit: {}".format(len(explicit_values)), linestyle="dashed")
        plt.loglog(implicit_values, label="Implicit: {}".format(len(implicit_values)))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Residual')
        plt.grid()
        plt.title('Convergence History of transient equation with grid size {}'.format(x_value))
        plt.legend()
        plt.savefig("homework1-task2-plots/error-convergence-transient-{}.png".format(x_value), format="png")
        plt.clf()


def plot_grid_density_errors(solver_results):
    for solver in solver_results.keys():

        mae = []
        for compare in [5, 10]:

            # Calculate indices for fine and coarse grid that have same value in range 0, 1
            fine_idx = []
            compare_idx = []
            for pos_fine in range(18):
                for pos_compare in range(compare + 3):
                    if (pos_compare * (1 / 17) == (pos_fine * (1 / (compare + 2)))):
                        fine_idx.append(pos_fine)
                        compare_idx.append(pos_compare)

            # Calculate absolute error for all combinations of grid indices
            absolute_errors = []
            for i in fine_idx:
                for j in fine_idx:
                    for x in compare_idx:
                        for y in compare_idx:
                            absolute_errors.append(
                                abs(solver_results[solver][15][i][j] - solver_results[solver][compare][x][y]))

            mae.append(np.mean(absolute_errors))

        plt.plot([5, 10], mae, label=solver)

    # Edit plot settings
    plt.xlabel("Delta X")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.title("Error of matrix solvers with different grid densities")
    plt.savefig("homework1-task2-plots/grid-density-errors.png", format="png")
    plt.clf()


def run_steady_main_logic(solver, size, gamma):

    # Initialize variables and grid
    nx = ny = size

    T = np.zeros((nx + 2, ny + 2))

    dx = 1 / nx
    dy = 1 / ny

    xmax = ymax = 1
    xmin = ymin = 0

    rho = 1.0

    # Initialize variables for finite volume calculation
    xnode = np.linspace(xmin, xmax, nx + 1)
    xcell = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    xcell = np.concatenate(([xmin], xcell, [xmax]))

    ynode = np.linspace(ymin, ymax, ny + 1)
    ycell = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
    ycell = np.concatenate(([ymin], ycell, [ymax]))

    # Initialize convection velocity
    u = np.zeros((nx + 1, nx + 1))
    v = np.zeros((ny + 1, ny + 1))

    for i in range(nx + 1):
        for j in range(ny + 1):
            u[i, j] = xnode[i]

    for i in range(nx + 1):
        for j in range(ny + 1):
            v[i, j] = -ynode[j]

    # Calculate finite volume
    a_e, a_w, a_n, a_s = calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho)

    # Calculate with Gauss Seidel or SOR method
    if solver == "gauss-seidel":
        T, convergence = calculate_gauss_seidel(T, nx, ny, ycell, a_e, a_w, a_n, a_s)
    elif solver == "sor":
        T, convergence = calculate_sor(T, nx, ny, ycell, a_e, a_w, a_n, a_s)
    else:
        raise TypeError('Solver not valid!')

    # Plot contour
    plot_contour(xcell, ycell, T, solver, size, nx, ny, ycell, ymax, gamma)

    return T, convergence


def run_transient_main_logic(solver, size):
    # Initialize variables and grid
    alpha = 1.172e-5  # thermal diffusivity of steel with 1% carbon

    nx = ny = size

    B = np.zeros(nx * ny)
    T = np.zeros((nx + 2, ny + 2))

    dx = 1 / nx
    dy = 1 / ny

    xmax = ymax = 1
    xmin = ymin = 0

    gamma = 0.1
    rho = 1.2

    dk = 0.0025
    max_k = 1000

    # Courant-Friedrichs-Lewy (CFL) condition to ensure stability
    cfl_x = alpha * dk / (dx ** 2)
    cfl_y = alpha * dk / (dy ** 2)

    if cfl_x > 0.5 or cfl_y > 0.5:
        raise TypeError('Unstable Solution!')

    # Initialize variables for finite volume calculation
    xnode = np.linspace(xmin, xmax, nx + 1)
    xcell = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    xcell = np.concatenate(([xmin], xcell, [xmax]))

    ynode = np.linspace(ymin, ymax, ny + 1)
    ycell = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
    ycell = np.concatenate(([ymin], ycell, [ymax]))

    # Initialize convection velocity
    u = np.zeros((nx + 1, nx + 1))
    v = np.zeros((ny + 1, ny + 1))

    for i in range(nx + 1):
        for j in range(ny + 1):
            u[i, j] = xnode[i]

    for i in range(nx + 1):
        for j in range(ny + 1):
            v[i, j] = -ynode[j]

    # Calculate with explicit or implicit Euler method
    if solver == "explicit":
        a_e, a_w, a_n, a_s = calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho)
        T, convergence = calculate_explicit_euler(T, max_k, dk, nx, ny, ymax, xnode, ynode, ycell, a_e, a_w, a_n, a_s,
                                                  rho)
    elif solver == "implicit":
        T, convergence = calculate_implicit_euler(B, T, max_k, dk, nx, ny, ymax, rho, xnode, xcell, ynode, ycell, gamma,
                                                  u, v)
    else:
        raise TypeError('Solver not valid!')

    # Plot contour
    plot_contour(xcell, ycell, T, solver, size, nx, ny, ycell, ymax)

    return T, convergence


def config_steady_search_space():
    # Initialize variables
    x_values = [5, 10, 15]
    solver_results = {"gauss-seidel (gamma=0.01)": {}, "gauss-seidel (gamma=0.001)": {},
                      "sor (gamma=0.01)": {}, "sor (gamma=0.001)": {}}

    # Iterate over variable combinations
    for size in x_values:
        for gamma in [0.01, 0.001]:
            conv_results = {}
            for solver in ["gauss-seidel", "sor"]:
                print("----------------------")
                print("Current solver is {} with grid size of {} gamma={}".format(solver, size, gamma))
                print("----------------------")

                T, convergence = run_steady_main_logic(solver, size, gamma)
                solver_results[solver + " (gamma={})".format(gamma)][size] = T
                conv_results[solver] = convergence

                print(" ")

            plot_error_convergence(conv_results, size, gamma)

    return solver_results


def config_transient_search_space():
    # Initialize variables
    x_values = [5, 10, 15]
    solver_results = {"explicit": {}, "implicit": {}}

    # Iterate over variable combinations
    for size in x_values:
        conv_results = {}
        for solver in ["explicit", "implicit"]:
            print("----------------------")
            print("Current solver is {} with grid size of {}".format(solver, size))
            print("----------------------")

            T, convergence = run_transient_main_logic(solver, size)
            solver_results[solver][size] = T
            conv_results[solver] = convergence

            print(" ")

        plot_error_convergence(conv_results, size)

    return solver_results


if __name__ == '__main__':
    # if the demo_folder directory is not present then create it
    if not os.path.exists("homework1-task2-plots"):
        os.makedirs("homework1-task2-plots")

    # Solve steady and transient equation
    steady_solver_results = config_steady_search_space()
    transient_solver_results = config_transient_search_space()

    # Merge results of solutions
    merged_solver_results = steady_solver_results.copy()
    merged_solver_results.update(transient_solver_results)

    # Plot grid density errors with finest grid as reference
    plot_grid_density_errors(merged_solver_results)
