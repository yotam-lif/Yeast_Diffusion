import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

# Define parameters
resolution_density = 2 * 10 ** -2
r = 0.1
N_max = 110
omega = 0.6  # Fixed omega value
L_x = 20  # Define L_x for the x axis
L_z = 40  # Define L_z for the z axis

# Create directory for saving figures
figs_dir = 'figs_analytic'
os.makedirs(figs_dir, exist_ok=True)


def k(n, L_x):
    return np.pi * n / (2 * L_x)


# Define the solution C(x, z)
def C(_x, _z, _L_x, _L_z, _r, _omega, _N_max):
    sum_result = np.zeros_like(_x, dtype=float)
    for n in range(_N_max + 1):
        kodd = k(2 * n + 1, _L_x)
        numerator = np.sin(kodd * _x) * (np.cosh(kodd * _z) - np.sinh(kodd * _z) * np.tanh(kodd * _L_z))
        denominator = kodd * (_omega + kodd * np.tanh(kodd * _L_z))
        sum_result += numerator / denominator
    return (_r / (2 * _omega)) - (_r / _L_x) * sum_result


def plot_heatmap_with_vectors(C_values, omega, L_x, L_z):
    """Plot and save a heatmap of the current grid with vector field."""
    dz, dx = np.gradient(C_values)
    dz, dx = -dz, -dx  # Invert the gradient to show flow direction
    x = np.linspace(-L_x, L_x, C_values.shape[1])
    z = np.linspace(0, L_z, C_values.shape[0])
    X, Z = np.meshgrid(x, z)

    # Reduce the density of vectors for better visibility
    step = max(1, len(x) // 12)

    plt.figure(figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])
    plt.imshow(C_values, cmap='viridis', aspect='auto', origin='lower', extent=[-L_x, L_x, 0, L_z])
    cbar = plt.colorbar(label='Value')
    cbar.ax.yaxis.label.set_size(16)
    # cbar.ax.yaxis.label.set_weight('bold')
    cbar.ax.yaxis.label.set_family('DejaVu Serif')
    plt.quiver(X[::step, ::step], Z[::step, ::step], dx[::step, ::step], dz[::step, ::step], color='white')
    plt.title(f'$C(x,z)$ steady state ; $\\Omega={omega:.1f}$', fontsize=16, weight='bold')
    plt.xlabel('$x$', fontsize=16, weight='bold')
    plt.ylabel('$z$', fontsize=16, weight='bold')
    plt.savefig(os.path.join(figs_dir, f'omega_{omega:.1f}.png'), dpi=200)
    plt.close()


# Main loop for a single L_x and L_z value
num_x = int(L_x / resolution_density)
num_z = int(L_z / resolution_density)

# Create the grid
X = np.linspace(-L_x, L_x, num_x)
Z = np.linspace(0, L_z, num_z)
X, Z = np.meshgrid(X, Z)

# Compute the solution for C(x, z)
C_values = C(X, Z, L_x, L_z, r, omega, N_max)

# Plot heatmap with vectors
plot_heatmap_with_vectors(C_values, omega, L_x, L_z)
