from Funcs import C_abs_z
import os
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
resolution_density = 2 * 10 ** -2
r = 1.0
N_max = 100
omega = 1.0  # Fixed omega value
L_x = 20  # Define L_x for the x axis
x_size = 10
z_size = 10

# Create directory for saving figures
figs_dir = '../figs'
os.makedirs(figs_dir, exist_ok=True)
fig_path = os.path.join(figs_dir, f'absorbing_bc.png')

num_x = int(x_size / resolution_density)
num_z = int(z_size / resolution_density)
# Create the grid
X = np.linspace(-L_x, L_x, num_x)
Z = np.linspace(0, z_size, num_z)
X, Z = np.meshgrid(X, Z)

# Compute the solution for C(x, z)
C_values = C_abs_z(X, Z, L_x, r, omega, N_max)

C_z_zero = C_values[:, 0]
plt.plot(X, C_z_zero)
plt.xlabel('x')
plt.ylabel('C(x, 0)')
plt.title(f'$C(x, 0)$ for infinite $L_Z; L_x={L_x:.1f}, r/D={r:.1f}, \\Omega/D$={omega:.1f}$')
