import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho):
    a_w = np.zeros((nx, ny))
    a_e = np.zeros((nx, ny))
    a_n = np.zeros((nx, ny))
    a_s = np.zeros((nx, ny))

    for i in range(1, nx + 1):
        dx = xnode[i] - xnode[i - 1]
        for j in range(1, ny + 1):
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


def calculate_gauss_seidel(T, nx, ny, ycell, a_e, a_w, a_n, a_s):

    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)

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
                T_old = T[i, j]
                T_new = (a_e[i - 1, j - 1] * Te + a_w[i - 1, j - 1] * Tw + a_n[i - 1, j - 1] * Tn + a_s[
                    i - 1, j - 1] * Ts) / a_p[i - 1, j - 1]
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
    omega = 1.5

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

    for k in np.arange(0, max_k):

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        T_decay = T.copy()
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
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
                t_p = T_decay[i, j]
                T[i, j] = a_e[i - 1, j - 1] * t_e + a_w[i - 1, j - 1] * t_w + a_n[i - 1, j - 1] * t_n + a_s[i - 1, j - 1] * t_s + (
                        ap0 - a_e[i - 1, j - 1] - a_w[i - 1, j - 1] - a_n[i - 1, j - 1] - a_s[i - 1, j - 1]) * t_p
                T[i, j] = T[i, j] / a_p[i - 1, j - 1]

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)

    return T


def calculate_implicit_euler(B, T, max_k, dk, nx, ny, ymax, rho, xnode, xcell, ynode, ycell, gamma, u, v):
    ap0 = np.zeros((nx, ny))
    a_p = np.zeros((nx, ny))

    for k in np.arange(0, max_k):
        Told = T.copy()
        a_e, a_w, a_n, a_s = calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho)

        for i in range(1, nx + 1):
            dx = xnode[i] - xnode[i - 1]
            for j in range(1, ny + 1):
                dy = ynode[j] - ynode[j - 1]
                ap0[i - 1, j - 1] = rho * dx * dy / dk
                a_p[i - 1, j - 1] = a_w[i - 1, j - 1] + a_e[i - 1, j - 1] + a_n[i - 1, j - 1] + a_s[i - 1, j - 1] + ap0[i - 1, j - 1]
                B[(j - 2) * nx + i - 1] = ap0[i - 1, j - 1] * Told[i - 1, j - 1]

        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                B[(j - 2) * nx + i - 1] = ap0[i - 1, j - 1] * Told[i, j]

        for i in range(0, nx):
            for j in range(0, ny):
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

        tol = 1.e-5
        iter_max = 10000
        iter = 0
        ea = 0.0

        T = np.zeros((nx + 2, ny + 2))
        iter = 0

        while True:
            eamax = 0.0
            for i in range(1, nx + 1):
                for j in range(1, ny + 1):
                    if i == 1:
                        t_w = 1 - ycell[j] / ymax
                    else:
                        t_w = T[i - 1, j]
                    if i == nx:
                        t_e = T[i, j]
                    else:
                        t_e = T[i + 1, j]
                    if j == 1:
                        t_s = T[i, j]
                    else:
                        t_s = T[i, j - 1]
                    if j == ny:
                        t_n = 0
                    else:
                        t_n = T[i, j + 1]

                    T_old = T[i, j]
                    T_new = (a_e[i - 1, j - 1] * t_e + a_w[i - 1, j - 1] * t_w + a_n[i - 1, j - 1] * t_n + a_s[
                        i - 1, j - 1] * t_s + B[(j - 2) * nx + i - 1]) / a_p[i - 1, j - 1]
                    T[i, j] = T_old + (T_new - T_old)
                    if T[i, j] != 0:
                        ea = abs(T[i, j] - T_old) / T[i, j] * 100
                    if ea >= eamax:
                        eamax = ea

            iter += 1
            if iter >= iter_max or eamax <= tol:
                break

    return T


def plot_contour(X, Y, T, solver, size, nx, ny, ycell, ymax):

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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(X, Y, T[:, :], levels=20, cmap='viridis_r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(cs, ticks=[i for i in np.arange(0.0, 1.05, 0.05)], label="Phi")
    plt.savefig("homework1-task2-plots/contour-{}-{}.png".format(solver, size), format="png")
    plt.clf()


def run_steady_main_logic(solver, size):
    alpha = 1.172e-5  # thermal diffusivity of steel with 1% carbon

    nx = ny = size

    T = np.zeros((nx + 2, ny + 2))

    dx = 1 / nx
    dy = 1 / ny

    xmax = ymax = 1
    xmin = ymin = 0

    gamma = 0.01
    rho = 1.0

    xnode = np.linspace(xmin, xmax, nx + 1)
    xcell = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    xcell = np.concatenate(([xmin], xcell, [xmax]))

    ynode = np.linspace(ymin, ymax, ny + 1)
    ycell = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
    ycell = np.concatenate(([ymin], ycell, [ymax]))

    u = np.zeros((nx + 1, nx + 1))
    v = np.zeros((ny + 1, ny + 1))

    for i in range(nx + 1):
        for j in range(ny + 1):
            u[i, j] = xnode[i]

    for i in range(nx + 1):
        for j in range(ny + 1):
            v[i, j] = -ynode[j]

    a_e, a_w, a_n, a_s = calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho)

    if solver == "gauss-seidel":
        T, convergence = calculate_gauss_seidel(T, nx, ny, ycell, a_e, a_w, a_n, a_s)
    elif solver == "sor":
        T, convergence = calculate_sor(T, nx, ny, ycell, a_e, a_w, a_n, a_s)
    else:
        raise TypeError('Solver not valid!')

    plot_contour(xcell, ycell, T, solver, size, nx, ny, ycell, ymax)


def run_unsteady_main_logic(solver, size):
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

    xnode = np.linspace(xmin, xmax, nx + 1)
    xcell = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
    xcell = np.concatenate(([xmin], xcell, [xmax]))

    ynode = np.linspace(ymin, ymax, ny + 1)
    ycell = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
    ycell = np.concatenate(([ymin], ycell, [ymax]))

    u = np.zeros((nx + 1, nx + 1))
    v = np.zeros((ny + 1, ny + 1))

    for i in range(nx + 1):
        for j in range(ny + 1):
            u[i, j] = xnode[i]

    for i in range(nx + 1):
        for j in range(ny + 1):
            v[i, j] = -ynode[j]

    if solver == "explicit":
        a_e, a_w, a_n, a_s = calculate_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho)
        T = calculate_explicit_euler(T, max_k, dk, nx, ny, ymax, xnode, ynode, ycell, a_e, a_w, a_n, a_s, rho)
    elif solver == "implicit":
        T = calculate_implicit_euler(B, T, max_k, dk, nx, ny, ymax, rho, xnode, xcell, ynode, ycell, gamma, u, v)
    else:
        raise TypeError('Solver not valid!')

    plot_contour(xcell, ycell, T, solver, size, nx, ny, ycell, ymax)


def config_steady_search_space():

    # Initialize variables
    x_values = [10, 20]
    solver_results = {}

    # Iterate over variable combinations
    for solver in ["gauss-seidel", "sor"]:
        solver_results[solver] = {}
        for size in x_values:
            print("----------------------")
            print("Current solver is {} with grid size of {}". format(solver, size))
            print("----------------------")

            run_steady_main_logic(solver, size)

            print(" ")


def config_unsteady_search_space():
    # Initialize variables
    x_values = [10, 20]
    solver_results = {}

    # Iterate over variable combinations
    for solver in ["explicit", "implicit"]:
        solver_results[solver] = {}
        for size in x_values:
            print("----------------------")
            print("Current solver is {} with grid size of {}".format(solver, size))
            print("----------------------")

            run_unsteady_main_logic(solver, size)

            print(" ")


if __name__ == '__main__':
    #config_steady_search_space()
    config_unsteady_search_space()
