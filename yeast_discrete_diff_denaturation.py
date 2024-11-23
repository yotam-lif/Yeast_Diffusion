import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

# Define parameters
r = 0.1
dt = 0.2
itrs = 70 * (10 ** 3)
L_x = 40
Omega = 0.6
omega = 0.1

# Create directory for saving figures
figs_dir = 'figs'
os.makedirs(figs_dir, exist_ok=True)


def Theta(x):
    """Heaviside step function."""
    if x < 0.0:
        return 1.0
    elif x == 0.0:
        return 0.5
    else:
        return 0.0


def enforce_boundary_conditions(C):
    """Apply boundary conditions to the grid."""
    C[:, 0] = C[:, 1]
    C[:, -1] = C[:, -2]
    C[-1, :] = C[-2, :]
    C[0, :] = C[1, :]


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


def flux(C):
    """Enforce flux boundary condition at z=0."""
    flux = np.zeros_like(C)
    Theta_vec = np.vectorize(Theta)
    flux[0, :] = r * Theta_vec(np.linspace(-L_x, L_x, C.shape[1])) - Omega * C[0, :]
    return flux


def plot_heatmap_with_vectors(C):
    """Plot and save a heatmap of the current grid with vector field."""
    dz, dx = np.gradient(C)
    dz, dx = -dz, -dx  # Invert the gradient to show flow direction
    x = np.linspace(-L_x, L_x, C.shape[1])
    z = np.linspace(0, 2 * L_x, C.shape[0])
    X, Z = np.meshgrid(x, z)

    # Reduce the density of vectors for better visibility
    step = max(1, len(x) // 12)

    plt.figure(figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])
    plt.imshow(C, cmap='viridis', aspect='auto', origin='lower', extent=[-L_x, L_x, 0, 2 * L_x])
    cbar = plt.colorbar(label='Value')
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.yaxis.label.set_family('DejaVu Serif')
    plt.quiver(X[::step, ::step], Z[::step, ::step], dx[::step, ::step], dz[::step, ::step], color='white')
    plt.title(f'$C(x,z)$ steady state', fontsize=16, weight='bold')
    plt.xlabel('$x$', fontsize=16, weight='bold')
    plt.ylabel('$z$', fontsize=16, weight='bold')
    plt.savefig(os.path.join(figs_dir, 'denaturation.png'), dpi=200)
    plt.close()


def plot_1d_lines(data):
    """Plot all 1D plots together."""
    plt.figure(figsize=(10, 6))
    plt.style.use(['science', 'no-latex'])
    x_values = np.linspace(-L_x, L_x, len(data))
    normalized_data = data / (r / Omega)
    plt.plot(x_values, normalized_data, label=f'$\\Omega={omega:.1f}$')
    plt.axhline(y=0.5, linestyle='--', label='$0.5$', color='b')
    plt.axhline(y=1.0, linestyle='--', label='$1$', color='r')
    plt.title(f'$r={r}$; $L_x={L_x}$', fontsize=16, weight='bold')
    plt.xlabel('$x$', fontsize=16, weight='bold')
    plt.ylabel('$C(x, 0) / (r / \\Omega)$', fontsize=16, weight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'z0_line_plot.png'), dpi=200)
    plt.close()


# Set initial grid with 0
plots_data = {}

# Main loop for different omega values
n_x = L_x * 2
n_z = n_x + 10
# Reset grid
C = np.zeros((n_z, n_x))

for I in range(itrs + 1):
    enforce_boundary_conditions(C)
    C += flux(C)
    C += (nab_sq(C) - omega * C) * dt

plot_heatmap_with_vectors(C)

# Plot all 1D plots together
plot_1d_lines(C[0, :].copy())