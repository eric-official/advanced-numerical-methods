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

                u = xnode[i]
                v = -ynode[j]

                # Convection term m attributes
                me = rho * u * dy
                mw = rho * u * dy
                mn = rho * v * dx
                ms = rho * v * dx

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
                a_e = ace + ade
                a_w = acw + adw
                a_n = acn + adn
                a_s = acs + ads
                a_p = acp + adp

                if i == 1:
                    Tw = 1 - ycell[j]
                else:
                    Tw = T[i - 1, j]

                if i == nx:
                    Te = T[i,j]
                else:
                    Te = T[i + 1, j]

                if j == 1:
                    Ts = T[i, j]
                else:
                    Ts = T[i, j - 1]

                if j == ny:
                    Tn = 0
                else:
                    Tn = T[i, j + 1]

                # Calculate T
                tij = (a_e * Te + a_w * Tw + a_n * Tn + a_s * Ts) / a_p
                T[i, j] = alpha * tij + T_decay[i, j]

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
                u = xnode[i]
                v = -ynode[j]

                # Convection term m attributes
                me = rho * u * dy
                mw = rho * u * dy
                mn = rho * v * dx
                ms = rho * v * dx

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
                a_e = ace + ade
                a_w = acw + adw
                a_n = acn + adn
                a_s = acs + ads
                a_p = acp + adp

                # Calculate T
                tij = (a_e * T_decay[i + 1, j] + a_w * T[i - 1, j] + a_n * T_decay[i, j + 1] + a_s * T[
                    i, j - 1]) / a_p
                T[i, j] = (1 - omega) * T_decay[i, j] + omega * (alpha * tij + T[i, j])

        # Update loop parameters
        current_error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        convergence = np.append(convergence, current_error)
        T_decay = np.copy(T)
        k += 1

    return T, convergence


def calculate_explicit_euler(T, max_k, dk, nx, ny, xnode, ynode, xcell, ycell, dx, dy, alpha):

    # Initialize loop parameters
    current_error = decay_error = 10
    k = 0
    convergence = np.array([], float)
    T_decay = np.copy(T)

    gamma = 0.001
    rho = 1.0

    for k in range(0, max_k - 1):

        # Print error every log10 steps
        if int(math.log10(current_error)) != int(math.log10(decay_error)):
            print("Current error", current_error)
        decay_error = current_error

        # Calculate with Gauss Seidel method
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):
                u = xnode[i]
                v = -ynode[j]

                # Convection term m attributes
                me = rho * u * dy
                mw = rho * u * dy
                mn = rho * v * dx
                ms = rho * v * dx

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
                a_e = ace + ade
                a_w = acw + adw
                a_n = acn + adn
                a_s = acs + ads
                a_p = acp + adp

                # Calculate T
                dv = (xnode[i] - xnode[i - 1]) * (ynode[j] - ynode[j - 1])
                ai_phi = a_p * T_decay[k - 1, i, j] + a_e * T_decay[k - 1, i + 1, j] + a_w * T_decay[k - 1, i - 1, j] \
                         + a_n * T_decay[k - 1, i, j + 1] + a_s * T_decay[k - 1, i, j - 1]
                T[k, i, j] = T_decay[k - 1, i, j] + (dk / dv / gamma) * (ai_phi - a_p * T_decay[k - 1, i, j])

        if k != 0:
            current_error = abs(np.linalg.norm(T[k, :, :]) - np.linalg.norm(T[k - 1, :, :]))
        convergence = np.append(convergence, current_error)

    return T, convergence


def plot_contour(X, Y, T, solver, size):
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
        T_solved, convergence = calculate_explicit_euler(T_init, max_k, dk, nx, ny, xnode, ynode, xcell, ycell, dx, dy, alpha)
    else:
        raise TypeError('Solver not valid!')

    if solver in ["gauss-seidel", "sor"]:
        plot_contour(X, Y, T_solved, solver, size)
    elif solver in ["explicit", "implicit"]:
        plot_contour(X, Y, T_solved[max_k - 1, :, :], solver, size)
    else:
        raise TypeError('Solver not valid!')


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


# Diffusion term with central difference
"""ade = -(2 * gamma) / ((X[i + 1] - X[i - 1]) * (X[i + 1] - X[i]))
adw = -(2 * gamma) / ((X[i + 1] - X[i - 1]) * (X[i] - X[i - 1]))
adn = -(2 * gamma) / ((Y[j + 1] - Y[j - 1]) * (Y[j + 1] - Y[j]))
ads = -(2 * gamma) / ((Y[j + 1] - Y[j - 1]) * (Y[j] - Y[j - 1]))
adp = -(ade + adw + adn + ads)"""

# Convection term with central difference
"""ace = -(roh * X[i]) / (X[i + 1] - X[i - 1])
acw = (roh * X[i]) / (X[i + 1] - X[i - 1])
acn = -(roh * Y[i]) / (Y[i + 1] - Y[i - 1])
acs = (roh * Y[i]) / (Y[i + 1] - Y[i - 1])
acp = -(ace + acw + acn + acs)"""

#part1 = (1 / (2 * dx * dy)) * (T_decay[i + 1, j] + T_decay[i - 1, j] + T_decay[i, j + 1] + T_decay[i, j - 1])
#part2 = (roh * X[i] * dy / (2 * dx)) * (T_decay[i + 1, j] - T_decay[i - 1, j])
#part3 = (roh * -Y[i] * dx / (2 * dy)) * (T_decay[i, j + 1] - T_decay[i, j - 1])
""""a = (T_decay[i + 1, j] - 2 * T_decay[i, j] + T[i - 1, j]) / dx**2
b = (T_decay[i, j + 1] - 2 * T_decay[i, j] + T[i, j - 1]) / dy**2
part2 = a * X[i]
part3 = b * -Y[i]
part4 = alpha * gamma * (a + b)
T[i, j] = part2 + part3 - part4"""


""" video approach
ace = gamma - rho * u * dx / 2
acw = gamma + rho * u * dx / 2
acn = gamma - rho * v * dy / 2
acs = gamma + rho * v * dy / 2
acp = rho * u * dx / 2 - rho * u * dx / 2 + rho * v * dy / 2 - rho * v * dy / 2 + 4 * gamma
T[i, j] = (ace * T[i + 1, j] + acw * T[i - 1, j] + acn * T_decay[i, j + 1] + acs * T_decay[i, j - 1]) / acp
"""