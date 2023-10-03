import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
alpha = 1.172E-5  # thermal diffusivity of steel with 1% carbon
# k = 9.7e-5 # thermal diffusivity of aluminium
L = 1.0  # length
W = 1.0  # width

# Numerical parameters
nx = 50  # number of points in x direction
ny = 50  # number of points in y direction
dk = 0.01  # time step
max_k = 100  # final time

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

T = np.zeros((nx, ny, int(max_k / dk)))

# Boundary conditions set-up
for i in range(0, nx):
    T[i, 0, 0] = lower_t
    T[i, ny - 1, 0] = upper_t

for j in range(0, ny):
    T[0, j, 0] = left_t
    T[nx - 1, j, 0] = right_t

# Generate 2D mesh
X = np.linspace(0, L, nx, endpoint=True)
Y = np.linspace(0, W, ny, endpoint=True)
X, Y = np.meshgrid(X, Y)

# Main time-loop
for k in range(0, int(max_k / dk) - 1):
    for i in range(1, (nx - 1)):
        for j in range(1, (ny - 1)):
            a = (T[i + 1, j, k] - 2 * T[i, j, k] + T[i - 1, j, k]) / dx ** 2  # d2dx2
            b = (T[i, j + 1, k] - 2 * T[i, j, k] + T[i, j - 1, k]) / dy ** 2  # d2dy2
            T[i, j, k + 1] = alpha * (a + b) + T[i, j, k]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, T[:, :, int(max_k / dk) - 1], cmap='gist_rainbow_r', edgecolor='none')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('T [°]')
plt.show()
