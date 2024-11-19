import numpy as np
import os
from Funcs import plot_ref_grid, C_ref_z

resolution_density = 2 * 10 ** -2
r = 1.0
N_max = 100
omega = 1.0  # Fixed omega value
L_x = 20  # Define L_x for the x axis
L_z = 40  # Define L_z for the z axis
x_size = L_x
z_size = L_z

# Create directory for saving figures
figs_dir = '../figs'
os.makedirs(figs_dir, exist_ok=True)
fig_path = os.path.join(figs_dir, f'analytic_reflecting_bc.png')

# Main loop for a single L_x and L_z value
num_x = int(L_x / resolution_density)
num_z = int(L_z / resolution_density)

# Create the grid
X = np.linspace(-L_x, L_x, num_x)
Z = np.linspace(0, L_z, num_z)
X, Z = np.meshgrid(X, Z)

# Compute the solution for C(x, z)
C_values = C_ref_z(X, Z, L_x, L_z, r, omega, N_max)

# Plot heatmap with vectors
plot_ref_grid(C_values, r, omega, L_x, L_z, x_size, z_size, fig_path)
