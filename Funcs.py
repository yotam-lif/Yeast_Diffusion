import numpy as np
import matplotlib.pyplot as plt
import scienceplots


# Heaviside step function
def theta(x: np.ndarray) -> np.ndarray:
    return 0.5 * (np.sign(x) + 1)


def k(n, L_x):
    return np.pi * n / (2 * L_x)


def C_abs_z(_x, _z, _L_x, _r, _omega, _N_max):
    sum_result = np.zeros_like(_x, dtype=float)
    for n in range(_N_max + 1):
        kodd = k(2 * n + 1, _L_x)
        numerator = np.sin(kodd * _x)
        denominator = _omega + kodd
        sum_result += numerator / denominator
    return (_r /  _omega) * (theta(-_x) + sum_result/_L_x)


# Define the solution C(x, z)
def C_ref_z(_x, _z, _L_x, _L_z, _r, _omega, _N_max):
    sum_result = np.zeros_like(_x, dtype=float)
    for n in range(_N_max + 1):
        kodd = k(2 * n + 1, _L_x)
        numerator = np.sin(kodd * _x) * (np.cosh(kodd * _z) - np.sinh(kodd * _z) * np.tanh(kodd * _L_z))
        denominator = kodd * (_omega + kodd * np.tanh(kodd * _L_z))
        sum_result += numerator / denominator
    return (_r / (2 * _omega)) - (_r / _L_x) * sum_result


def plot_abs_grid(C_values, r, omega, L_x, x_size, z_size, fig_path):
    """Plot and save a heatmap of the current grid with vector field."""
    dz, dx = np.gradient(C_values)
    dz, dx = -dz, -dx  # Invert the gradient to show flow direction
    x = np.linspace(-x_size, x_size, C_values.shape[1])
    z = np.linspace(0, z_size, C_values.shape[0])
    X, Z = np.meshgrid(x, z)

    # Reduce the density of vectors for better visibility
    step = max(1, len(x) // 12)

    plt.figure(figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])
    plt.imshow(C_values, cmap='viridis', aspect='auto', origin='lower', extent=[-L_x, L_x, 0, z_size])
    plt.quiver(X[::step, ::step], Z[::step, ::step], dx[::step, ::step], dz[::step, ::step], color='white')
    plt.colorbar()
    plt.title(f'$C(x,z)$ steady state ; $\\Omega={omega:.1f} ; r={r:.1f} ; L_x={L_x:.1f}$', fontsize=16, weight='bold')
    plt.xlabel('$x$', fontsize=16, weight='bold')
    plt.ylabel('$z$', fontsize=16, weight='bold')
    plt.savefig(fig_path, dpi=300)
    plt.close()

def plot_ref_grid(C_values, r, omega, L_x, L_z, x_size, z_size, fig_path):
    """Plot and save a heatmap of the current grid with vector field."""
    dz, dx = np.gradient(C_values)
    dz, dx = -dz, -dx  # Invert the gradient to show flow direction

    # Create the full grid
    x_full = np.linspace(-L_x, L_x, C_values.shape[1])
    z_full = np.linspace(0, L_z, C_values.shape[0])

    # Determine the indices for the sublattice
    x_indices = np.where((x_full >= -x_size) & (x_full <= x_size))[0]
    z_indices = np.where((z_full >= 0) & (z_full <= z_size))[0]

    # Create the reduced grid
    x_reduced = x_full[x_indices]
    z_reduced = z_full[z_indices]
    X_reduced, Z_reduced = np.meshgrid(x_reduced, z_reduced)

    # Extract the sublattice from C_values
    C_values_reduced = C_values[np.ix_(z_indices, x_indices)]
    dz_reduced = dz[np.ix_(z_indices, x_indices)]
    dx_reduced = dx[np.ix_(z_indices, x_indices)]

    # Reduce the density of vectors for better visibility
    step = max(1, len(x_reduced) // 18)

    plt.figure(figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])
    plt.imshow(C_values_reduced, cmap='viridis', aspect='auto', origin='lower', extent=[-x_size, x_size, 0, z_size])
    plt.quiver(X_reduced[::step, ::step], Z_reduced[::step, ::step], dx_reduced[::step, ::step], dz_reduced[::step, ::step], color='white')
    plt.title(f'$C(x,z)$ steady state ; $\\Omega/D={omega:.1f} ; r/D={r:.1f} ; L_x={L_x:.1f} ; L_z={L_z:.1f}$', fontsize=16, weight='bold')
    plt.colorbar()
    plt.xlabel('$x$', fontsize=16, weight='bold')
    plt.ylabel('$z$', fontsize=16, weight='bold')
    plt.savefig(fig_path, dpi=300)
    plt.close()
