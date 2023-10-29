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


def initialize_unsteady_grid(max_k, nx, ny, lower_t, upper_t, left_t, right_t):
    T = np.zeros((max_k, nx, ny))

    # Boundary conditions set-up
    T[:, 0, :] = lower_t
    T[:, nx - 1, :] = upper_t

    T[:, :, 0] = left_t
    T[:, :, ny - 1] = right_t

    return T


def calculate_finite_volume(xnode, ynode, xcell, ycell, dx, dy, i, j, gamma, rho):
    ue = xnode[i]
    uw = xnode[i - 1]
    vn = (-1) * ynode[j]
    vs = (-1) * ynode[j - 1]

    # Convection term m attributes
    me = rho * ue * dy
    mw = rho * uw * dy
    mn = rho * vn * dx
    ms = rho * vs * dx

    # Convection term lambda attributes
    le = (xnode[i] - xcell[i]) / (xcell[i + 1] - xcell[i])
    lw = (xcell[i - 1] - xcell[i]) / (xcell[i - 1] - xcell[i])
    ln = (ycell[j] - ycell[j]) / (ycell[j + 1] - ycell[j])
    ls = (ycell[j - 1] - ycell[j]) / (ycell[j - 1] - ycell[j])

    # Convection term with central difference
    ace = me * le
    acw = mw * lw
    acn = mn * ln
    acs = ms * ls
    acp = -(ace + acw + acn + acs)

    # Diffusion term with central difference
    ade = -(gamma * dy) / (xcell[i + 1] - xcell[i])
    adw = -(gamma * dy) / (xcell[i] - xcell[i - 1])
    adn = -(gamma * dx) / (ycell[j + 1] - ycell[j])
    ads = -(gamma * dx) / (ycell[j] - ycell[j - 1])
    adp = -(ade + adw + adn + ads)

    # Add convection and diffusion terms
    a_e = ace - ade
    a_w = acw - adw
    a_n = acn - adn
    a_s = acs - ads
    a_p = acp - adp

    return a_e, a_w, a_n, a_s, a_p


def calculate_unsteady_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho, dk):
    a_w = np.zeros((nx, ny))
    a_e = np.zeros((nx, ny))
    a_n = np.zeros((nx, ny))
    a_s = np.zeros((nx, ny))
    a_p = np.zeros((nx, ny))

    for i in range(1, nx + 1):
        dx = xnode[i] - xnode[i - 1]
        for j in range(1, ny + 1):
            dy = ynode[j] - ynode[j - 1]
            delx_e = xcell[i + 1] - xcell[i]
            delx_w = xcell[i] - xcell[i - 1]
            dely_n = ycell[j + 1] - ycell[j]
            dely_s = ycell[j] - ycell[j - 1]
            a_w[i - 1, j - 1] = gamma * dy / delx_w - rho * u[i - 1, j] * dy * (-1)
            a_e[i - 1, j - 1] = gamma * dy / delx_e - rho * u[i, j] * dy
            a_n[i - 1, j - 1] = gamma * dx / dely_n - rho * v[i, j] * dx
            a_s[i - 1, j - 1] = gamma * dx / dely_s - rho * v[i, j - 1] * dx * (-1)
            ap0 = rho * dx * dy / dk
            a_p[i - 1, j - 1] = ap0

    return a_e, a_w, a_n, a_s, a_p, ap0


def calculate_gauss_seidel(T, nx, ny, xnode, ynode, xcell, ycell, dx, dy, alpha):

    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)
    T_decay = np.copy(T)

    gamma = 0.001
    rho = 1.0

    while current_error >= 1e-5:

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        # Calculate with Gauss Seidel method
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):

                a_e, a_w, a_n, a_s, a_p = calculate_finite_volume(xnode, ynode, xcell, ycell, dx, dy, i, j, gamma, rho)

                # cell value next to west boundary
                if i == 1:
                    tw = 1 - ycell[j]
                else:
                    tw = T[i - 1, j]

                # cell value next to east boundary
                if i == nx - 1:
                    te = T[i, j]
                else:
                    te = T[i + 1, j]

                # cell value next to south boundary
                if j == 1:
                    ts = T[i, j]
                else:
                    ts = T[i, j - 1]

                # cell value next to north boundary
                if j == ny - 1:
                    tn = 0
                else:
                    tn = T[i, j + 1]

                # Calculate T
                tij = (a_e * te + a_w * tw + a_n * tn + a_s * ts) / a_p
                T[i, j] = tij

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        T_decay = np.copy(T)
        k += 1

    return T, convergence


def calculate_sor(T, nx, ny, xnode, ynode, xcell, ycell, dx, dy, alpha):

    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)
    T_decay = np.copy(T)

    omega = 1.5
    gamma = 0.001
    rho = 1.0

    while current_error >= 1e-5:

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        # Calculate with SOR method
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):

                a_e, a_w, a_n, a_s, a_p = calculate_finite_volume(xnode, ynode, xcell, ycell, dx, dy, i, j, gamma, rho)

                # cell value next to west boundary
                if i == 1:
                    tw = 1 - ycell[j]
                else:
                    tw = T[i - 1, j]

                # cell value next to east boundary
                if i == nx - 1:
                    te = T_decay[i, j]
                else:
                    te = T_decay[i + 1, j]

                # cell value next to south boundary
                if j == 1:
                    ts = T[i, j]
                else:
                    ts = T[i, j - 1]

                # cell value next to north boundary
                if j == ny - 1:
                    tn = 0
                else:
                    tn = T_decay[i, j + 1]

                # Calculate T
                tij = (a_e * te + a_w * tw + a_n * tn + a_s * ts) / a_p
                T[i, j] = (1 - omega) * T_decay[i, j] + omega * (tij)

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        T_decay = np.copy(T)
        k += 1

    return T, convergence


def calculate_explicit_euler(T, max_k, dk, nx, ny, ymax, ycell, a_e, a_w, a_n, a_s, a_p, ap0):
    for k in np.arange(0, max_k, dk):
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
    return T


def calculate_implicit_euler(B, T, max_k, dk, nx, ny, ymax, rho, xnode, xcell, ynode, ycell, a_e, a_w, a_n, a_s, gamma, u, v):
    ap0 = np.zeros((nx, ny))
    a_p = np.zeros((nx, ny))

    for k in np.arange(0, max_k):
        Told = T.copy()
        a_e, a_w, a_n, a_s, _, _ = calculate_unsteady_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho, dk)

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


def plot_contour(X, Y, T, solver, size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contourf(X, Y, T[:, :], levels=20, cmap='viridis_r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(cs, ticks=[i for i in np.arange(0.0, 1.05, 0.05)], label="Phi")
    plt.savefig("homework1-task2-plots/contour-{}-{}.png".format(solver, size), format="png")
    plt.clf()


def plot_unsteady_contour(X, Y, T, solver, size, nx, ny, ycell, ymax):

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


def run_main_logic(solver, size):
    # Physical parameters
    alpha = 1.172e-5  # thermal diffusivity of steel with 1% carbon
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

    xnode = X[0]
    ynode = Y[0]

    xcell = xnode[0]
    xcell = np.append(xcell, [0.5 * (xnode[i + 1] + xnode[i]) for i in range(0, nx - 1)])
    xcell = np.append(xcell, xnode[nx - 1])
    ycell = xcell.copy()

    if solver in ["gauss-seidel", "sor"]:
        T_init = initialize_steady_grid(nx, ny, lower_t, upper_t, left_t, right_t)
    elif solver in ["explicit", "implicit"]:
        T_init = initialize_unsteady_grid(max_k, nx, ny, lower_t, upper_t, left_t, right_t)
    else:
        raise TypeError('Solver not valid!')

    if solver == "gauss-seidel":
        T_solved, convergence = calculate_gauss_seidel(T_init, nx, ny, xnode, ynode, xcell, ycell, dx, dy, alpha)
    elif solver == "sor":
        T_solved, convergence = calculate_sor(T_init, nx, ny, xnode, ynode, xcell, ycell, dx, dy, alpha)
    elif solver == "explicit":
        T_solved = calculate_explicit_euler(T_init, max_k, dk, nx, ny, xnode, ynode, xcell, ycell, dx, dy, alpha)
    else:
        raise TypeError('Solver not valid!')

    if solver in ["gauss-seidel", "sor"]:
        plot_contour(X, Y, T_solved, solver, size)
    elif solver in ["explicit", "implicit"]:
        plot_contour(X, Y, T_solved[max_k - 1, :, :], solver, size)
    else:
        raise TypeError('Solver not valid!')


def run_unsteady_main_logic(solver, size):
    nx = ny = size

    B = np.zeros(nx * ny)
    T = np.zeros((nx + 2, ny + 2))

    dx = 1 / nx
    dy = 1 / ny

    xmax = ymax = 1
    xmin = ymin = 0

    gamma = 0.1
    rho = 1.2

    dk = 0.02
    max_k = 10000

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

    a_e, a_w, a_n, a_s, a_p, ap0 = calculate_unsteady_finite_volume(nx, ny, xnode, ynode, xcell, ycell, u, v, gamma, rho, dk)
    if solver == "explicit":
        T = calculate_explicit_euler(T, max_k, dk, nx, ny, ymax, ycell, a_e, a_w, a_n, a_s, a_p, ap0)
    elif solver == "implicit":
        T = calculate_implicit_euler(B, T, max_k, dk, nx, ny, ymax, rho, xnode, xcell, ynode, ycell, a_e, a_w, a_n, a_s, gamma, u, v)
    else:
        raise TypeError('Solver not valid!')
    plot_unsteady_contour(xcell, ycell, T, solver, size, nx, ny, ycell, ymax)


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


def config_unsteady_search_space():
    # Initialize variables
    x_values = [5, 10]
    solver_results = {}

    # Iterate over variable combinations
    for solver in ["implicit"]:
        solver_results[solver] = {}
        for size in x_values:
            print("----------------------")
            print("Current solver is {} with grid size of {}".format(solver, size))
            print("----------------------")

            run_unsteady_main_logic(solver, size)

            print(" ")


if __name__ == '__main__':
    #config_search_space()
    config_unsteady_search_space()
