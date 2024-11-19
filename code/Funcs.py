import numpy as np
import matplotlib.pyplot as plt
import scienceplots


# Heaviside step function
def theta(x: np.ndarray) -> np.ndarray:
    return 0.5 * (np.sign(x) + 1)


def kn(n, L_x):
    return np.pi * (2*n + 1) / (2 * L_x)


def C_abs_z(_x, _z, _L_x, _r, _omega, _N_max):
    sum_result = np.zeros_like(_x, dtype=float)
    for n in range(_N_max + 1):
        kodd = kn(n, _L_x)
        numerator = np.sin(kodd * _x)
        denominator = _omega + kodd
        sum_result += numerator / denominator
    return (_r /  _omega) * (theta(-_x) + sum_result/_L_x)


# Define the solution C(x, z)
def C_ref_z(_x, _z, _L_x, _L_z, _r, _omega, _N_max):
    sum_result = np.zeros_like(_x, dtype=float)
    for n in range(_N_max + 1):
        k_n = kn(n, _L_x)
        numerator = np.sin(k_n * _x) * np.cosh(k_n * (_z - _L_z))
        denominator = k_n * (k_n * np.sinh(k_n *_L_z) + _omega * np.cosh(k_n * _L_z))
        sum_result += numerator / denominator
    return (_r / (2 * _omega)) - (_r / _L_x) * sum_result


def C_ref_z0(_x, _L_x, _L_z, _r, _omega, _N_max):
    sum_result = np.zeros_like(_x, dtype=float)
    for n in range(_N_max + 1):
        k_n = kn(n, _L_x)
        numerator = np.sin(k_n * _x)
        denominator = k_n * (k_n * np.tanh(k_n *_L_z) + _omega)
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

def plot_ref_grid(C_values, r, omega, L_x, L_z, x_size, z_size, fig_path, x_exclude=3.0, z_exclude=2.0):
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

    # Mask out the region to exclude arrows
    mask = (X_reduced >= -x_exclude) & (X_reduced <= x_exclude) & (Z_reduced >= 0) & (Z_reduced <= z_exclude)
    dx_reduced[mask] = 0
    dz_reduced[mask] = 0

    # Reduce the density of vectors for better visibility
    step = max(1, len(x_reduced) // 15)

    plt.figure(figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])
    plt.imshow(C_values_reduced, cmap='viridis', aspect='auto', origin='lower', extent=[-x_size, x_size, 0, z_size])
    plt.quiver(X_reduced[::step, ::step], Z_reduced[::step, ::step], dx_reduced[::step, ::step], dz_reduced[::step, ::step], color='white')
    plt.title(f'$C(x,z)$ steady state ; $\\Omega/D={omega:.1f} ; r/D={r:.1f} ; L_x={L_x:.1f} ; L_z={L_z:.1f}$', fontsize=16, weight='bold')
    plt.colorbar()
    plt.xlabel('$x$', fontsize=16, weight='bold')
    plt.ylabel('$z$', fontsize=16, weight='bold')
    plt.savefig(fig_path, dpi=600)
    plt.close()


def enforce_reflecting_bc(C):
    """Apply boundary conditions to the grid."""
    # x = L_x
    C[0, :] = C[1, :]
    # x = -L_x
    C[-1, :] = C[-2, :]
    # z = L_z
    C[:, -1] = C[:, -2]
    # z = 0
    C[:, 0] = C[:, 1]




def nab_sq(C):
    """Calculate the discrete Laplacian using NumPy's slicing."""
    laplacian = np.zeros_like(C)
    # Corners
    laplacian[0, 0] = C[1, 0] + C[0, 1] - 2 * C[0, 0]
    laplacian[0, -1] = C[1, -1] + C[0, -2] - 2 * C[0, -1]
    laplacian[-1, 0] = C[-2, 0] + C[-1, 1] - 2 * C[-1, 0]
    laplacian[-1, -1] = C[-2, -1] + C[-1, -2] - 2 * C[-1, -1]
    # Edges
    laplacian[0, 1:-1] = C[1, 1:-1] + C[0, :-2] + C[0, 2:] - 3 * C[0, 1:-1]
    laplacian[-1, 1:-1] = C[-2, 1:-1] + C[-1, :-2] + C[-1, 2:] - 3 * C[-1, 1:-1]
    laplacian[1:-1, 0] = C[:-2, 0] + C[2:, 0] + C[1:-1, 1] - 3 * C[1:-1, 0]
    laplacian[1:-1, -1] = C[:-2, -1] + C[2:, -1] + C[1:-1, -2] - 3 * C[1:-1, -1]
    # Interior
    laplacian[1:-1, 1:-1] = (
        C[:-2, 1:-1] + C[2:, 1:-1] + C[1:-1, :-2] + C[1:-1, 2:] - 4 * C[1:-1, 1:-1]
    )
    return laplacian

def flux(C, r, omega, L_x):
    """Enforce flux boundary condition at z=0."""
    flux = np.zeros_like(C)
    x = np.linspace(-L_x, L_x, C.shape[1])
    flux[0, :] = r * theta(-x) - omega * C[0, :]
    return flux


