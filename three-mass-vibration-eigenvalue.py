import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sympy import Matrix, symbols, solve
import os


def power_iteration(matrix, eigenvector, max_iterations=1000, tolerance=1e-10):
    n = matrix.shape[0]

    for i in range(max_iterations):
        # Perform matrix-vector multiplication
        matrix_times_vector = np.dot(matrix, eigenvector)

        # Update the eigenvector
        eigenvector_new = matrix_times_vector / np.linalg.norm(matrix_times_vector)

        # Check convergence
        if np.linalg.norm(eigenvector - eigenvector_new) < tolerance:
            break

        eigenvector = eigenvector_new

    # Calculate the eigenvalue
    eigenvalues = np.dot(eigenvector, np.dot(matrix, eigenvector)) / np.dot(eigenvector, eigenvector)

    return eigenvalues, eigenvector


# Find all eigenvalues using shifted power method
def shifted_power_iteration(matrix, num_eigenvalues=3, max_iterations=1000, tolerance=1e-10):
    n = matrix.shape[0]
    eigenvalues = []
    eigenvalue_prev = 0

    # Start with a random vector
    eigenvector = np.random.rand(n)

    for _ in range(num_eigenvalues):
        # Use power iteration to find the dominant eigenvalue and eigenvector
        eigenvalue, eigenvector = power_iteration(matrix, eigenvector, max_iterations, tolerance)

        # Store the found eigenvalue
        eigenvalues.append(eigenvalue + eigenvalue_prev)
        eigenvalue_prev = eigenvalue

        # Shift the matrix for the next iteration
        matrix = (matrix - eigenvalues[-1] * np.identity(n))
        eigenvector = eigenvector / np.linalg.norm(eigenvector)

    return np.array(eigenvalues)


def calculate_middle_eigenvalue(eigenvalues):
    # Calculate determinant of matrix with middle eigenvalue as a variable
    l = symbols(['l'])[0]
    matrix = Matrix([[-(50 / 1.5) - eigenvalues[0], 35 / 1.5, 0], [35 / 1.5, (-70 / 1.5) - l, 35 / 1.5],
                     [0, 35 / 1.5, -(50 / 1.5) - eigenvalues[1]]])
    determinant = matrix.det()

    # Solve for middle eigenvalue
    middle_eigenvalue = solve(determinant, l)[0]

    # Set middle eigenvalue
    eigenvalues.setflags(write=1)
    eigenvalues[2] = middle_eigenvalue
    eigenvalues = np.sort(eigenvalues)

    return eigenvalues


def qr_algorithm(matrix, max_iterations=1000, tolerance=1e-10):
    n = matrix.shape[0]
    eigenvalues = np.zeros(n, dtype=complex)

    for i in range(max_iterations):
        # QR decomposition
        q, r = np.linalg.qr(matrix)

        # Update the matrix with RQ
        matrix = np.dot(r, q)

        # Extract the eigenvalues
        eigenvalues_new = np.diag(matrix)

        # Check convergence
        if np.allclose(eigenvalues, eigenvalues_new, rtol=tolerance):
            break

        eigenvalues = eigenvalues_new

    return eigenvalues


def main(algorithm):
    # Define Array from equations of motion.
    A = np.array([[-(50 / 1.5), 35 / 1.5, 0], [35 / 1.5, -70 / 1.5, 35 / 1.5], [0, 35 / 1.5, -(50 / 1.5)]])  # 2 masses
    _, eigenvectors = np.linalg.eig(A)  # Find Eigenvalues and vectors.

    if algorithm == "Shifted Power Method":
        eigenvalues = shifted_power_iteration(A)  # Find Eigenvalues and vectors.
        eigenvalues = calculate_middle_eigenvalue(eigenvalues)  # Find middle eigenvalue
    elif algorithm == "QR Algorithm":
        eigenvalues = qr_algorithm(A)
    else:
        raise TypeError('Algorithm not valid!')

    eigenvalues = np.diag(eigenvalues)  # nxn array with eigenvalues along the diagonal
    omega = np.sqrt(np.diag(-eigenvalues))  # Get frequencies
    x0 = np.array([1, 1, 1])
    gam = np.linalg.inv(eigenvectors) @ x0

    # nxn array with coefficients of gamma along the diagonal
    g = np.diag(gam)
    t = np.arange(0, 20.2, 0.2)  # 1xM Time vector (for plotting)

    # cos(omega*t) is an nxM array with cos(w1*t),...,cos(wn*t) in rows
    x = eigenvectors @ g @ np.cos(np.outer(omega, t))  # Calculate output

    # Display pertinent information about the system
    print('A matrix\n', A)
    print('Eigenvalues\n', np.diag(eigenvalues))
    print('Eigenvectors (each column is an eigenvector)\n', eigenvectors)
    print(f'Frequencies, omega={", ".join(map(lambda x: f"{x:.2f}", omega))}')
    print(f'Initial Conditions, x(0)={", ".join(map(lambda x: f"{x:.2f}", x0))}')
    print(f'Unknown coefficients, gamma={", ".join(map(lambda x: f"{x:.2f}", gam))}')

    # Plot the output trajectories
    plt.figure(figsize=(10, 6))
    for i in range(A.shape[0]):
        plt.plot(t, x[i, :], label=f'Out_{i + 1}')

    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title(f"{algorithm}: Output Trajectories")
    plt.legend()
    plt.grid()
    plt.savefig(f'homework2-task2-plots/output trajectories-{algorithm}.png', format='png')
    plt.clf()

    # Plot the model shapes
    plt.figure(figsize=(10, 6))
    for i in range(A.shape[0]):
        plt.plot(eigenvectors[i, :], label=f'Mode_{i + 1}')

    plt.xlabel('Elements')
    plt.ylabel('Mode Amplitude')
    plt.title(f"{algorithm}: Modal Shapes")
    plt.legend()
    plt.grid()
    plt.savefig(f'homework2-task2-plots/modal shapes-{algorithm}.png', format='png')
    plt.clf()


if __name__ == '__main__':
    # if the demo_folder directory is not present then create it
    if not os.path.exists("homework2-task2-plots"):
        os.makedirs("homework2-task2-plots")

    for alg in ["Shifted Power Method", "QR Algorithm"]:
        main(alg)
