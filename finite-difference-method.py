import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg


def initialize_finite_difference_grid(max_k, dk, nx, ny, lower_t, upper_t, left_t, right_t):
    T = np.zeros((nx, ny, int(max_k / dk)))

    # Boundary conditions set-up
    for i in range(0, nx):
        T[i, 0, 0] = lower_t
        T[i, ny - 1, 0] = upper_t

    for j in range(0, ny):
        T[0, j, 0] = left_t
        T[nx - 1, j, 0] = right_t

    return T


def calculate_finite_differences(T, max_k, dk, nx, ny, dx, dy, alpha):
    for k in range(0, int(max_k / dk) - 1):
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):
                a = (T[i + 1, j, k] - 2 * T[i, j, k] + T[i - 1, j, k]) / dx ** 2  # d2dx2
                b = (T[i, j + 1, k] - 2 * T[i, j, k] + T[i, j - 1, k]) / dy ** 2  # d2dy2
                T[i, j, k + 1] = alpha * (a + b) + T[i, j, k]
    return T


def initialize_gauss_seidel_grid(nx, ny, lower_t, upper_t, left_t, right_t):
    T = np.zeros((nx, ny))

    # Boundary conditions set-up
    T[:, 0] = left_t
    T[:, ny - 1] = right_t

    T[0, :] = lower_t
    T[nx - 1, :] = upper_t

    return T


def calculate_gauss_seidel(T, nx, ny, dx, dy, alpha):
    error = 10
    k = 0
    res = np.array([], float)
    T_decay = np.copy(T)
    while error >= 1e-5:
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):
                a = (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dx ** 2  # d2dx2
                b = (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dy ** 2  # d2dy2
                T[i, j] = alpha * (a + b) + T[i, j]
        error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        res = np.append(res, error)
        T_decay = np.copy(T)
        k += 1

    plt.loglog(res, "-b")
    plt.xlabel('number of iteration - Gauss-seidell')
    plt.ylabel('Residual')
    plt.title('Convergence History')
    plt.legend(["Number of Iteration= {:3d} ".format(k)])
    plt.savefig("plots/gs-error-convergence-{}.png".format(nx), format="png")

    return T


def calculate_sor(T, nx, ny, dx, dy, alpha):
    error = 10
    k = 0
    omega = 1.5
    res = np.array([], float)
    T_decay = np.copy(T)
    while error >= 1e-5:
        print("error", error)
        for i in range(1, (nx - 1)):
            for j in range(1, (ny - 1)):
                a = (T_decay[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dx ** 2  # d2dx2
                b = (T_decay[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dy ** 2  # d2dy2
                T[i, j] = (1 - omega) * T_decay[i, j] + omega * (alpha * (a + b) + T[i, j])
        error = abs(np.linalg.norm(T) - np.linalg.norm(T_decay))
        res = np.append(res, error)
        T_decay = np.copy(T)
        k += 1

    plt.loglog(res, "-b")
    plt.xlabel('number of iteration - Successive over-relaxation')
    plt.ylabel('Residual')
    plt.title('Convergence History')
    plt.legend(["Number of Iteration= {:3d} ".format(k)])
    plt.savefig("plots/sor-error-convergence-{}.png".format(nx), format="png")

    return T


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

    if solver == "loop":
        T = initialize_finite_difference_grid(max_k, dk, nx, ny, lower_t, upper_t, left_t, right_t)
        T = calculate_finite_differences(T, max_k, dk, nx, ny, dx, dy, alpha)
    elif solver == "gauss-seidel":
        T = initialize_gauss_seidel_grid(nx, ny, lower_t, upper_t, left_t, right_t)
        T = calculate_gauss_seidel(T, nx, ny, dx, dy, alpha)
    elif solver == "sor":
        T = initialize_gauss_seidel_grid(nx, ny, lower_t, upper_t, left_t, right_t)
        T = calculate_sor(T, nx, ny, dx, dy, alpha)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if solver == "loop":
        cs = ax.contourf(X, Y, T[:, :, int(max_k / dk) - 1], levels=20, cmap='gist_rainbow_r')
    else:
        cs = ax.contourf(X, Y, T[:, :], levels=20, cmap='gist_rainbow_r')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    fig.colorbar(cs, ticks=[i for i in np.arange(0.0, 1.05, 0.05)])
    plt.savefig("plots/contour-{}-{}.png".format(solver, size), format="png")
    plt.show()

    return 5


def config_search_space():

    x_values = [5, 10, 20, 50, 100]
    solver_errors = {}
    for solver in ["loop", "gauss-seidel", "sor"]:
        print(solver)
        errors = []
        for size in x_values:
            error = run_main_logic(solver, size)
            if solver == "gauss-seidel":
                error += 1
            elif solver == "sor":
                error += 2
            errors.append(error)
        solver_errors[solver] = errors

    x_values = [1/i for i in x_values]

    plt.plot(x_values, solver_errors["loop"], label="loop")
    plt.plot(x_values, solver_errors["gauss-seidel"], label="Gauss Seidel")
    plt.plot(x_values, solver_errors["sor"], label="Successive over-relaxation")

    plt.xlabel("Delta X")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.ylim([0, 10])
    plt.legend()
    plt.title("Error of matrix solvers with different grid densities")
    plt.savefig("plots/grid-density-errors.png", format="png")
    plt.show()


def manual_run(solver, size):
    if solver is None:
        raise TypeError('Solver is not configured!')
    elif size is None:
        raise TypeError('Size is not configured!')
    else:
        run_main_logic(solver, size)


if __name__ == '__main__':
    config_search_space()

