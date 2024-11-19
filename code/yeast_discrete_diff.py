import numpy as np
import matplotlib.pyplot as plt
import os
import Funcs as Fs
import scienceplots

# Define parameters
r_values = [0.25]  # List of r values to iterate over
omega_values = [0.25]  # List of omega values to iterate over
dt = 0.05  # Time step
itrs = 20 * (10 ** 3)  # Number of iterations
L_x = 10  # Domain size in x
L_z = 10  # Domain size in z
res =  1 * 10 ** -2  # Spatial resolution
n_x = int(2 * L_x / res) + 1  # Number of grid points in x
n_z = int(L_z / res) + 1  # Number of grid points in z
N_max = 120  # Maximum number of terms in the series solution
plt.style.use(['science'])

# Create directory for saving figures
figs_dir = '../figs'
os.makedirs(figs_dir, exist_ok=True)


def main():
    # Define the spatial grid
    x = np.linspace(-L_x, L_x, n_x)

    # Initialize figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Iterate over all combinations of r and omega
    for r in r_values:
        for omega in omega_values:
            print(f'Processing r={r}, omega={omega}...')

            # Initialize the concentration grid to zero
            C = np.ones((n_z, n_x))

            # Perform the main iteration loop
            for itr in range(itrs + 1):
                # Update concentration with flux and diffusion
                Fs.enforce_reflecting_bc(C)
                C += Fs.flux(C, r, omega, L_x)
                C += Fs.nab_sq(C) * dt
                if itr % 1000 == 0:
                    print(f'Iteration {itr}')

            # Extract numerical solution at z=0
            C_num = C[0, :]

            # Compute analytical solution at z=0
            C_an = Fs.C_ref_z0(x, L_x, L_z, r, omega, N_max)

            # Plot analytical solution
            axs[0].plot(x, C_an, label=f'$r={r}, \\Omega={omega}$')

            # Plot numerical solution
            axs[1].plot(x, C_num, label=f'$r={r}, \\Omega={omega}$')

    # Configure Analytical Solution subplot
    axs[0].set_title('Analytical Solution', fontsize=18)
    axs[0].set_xlabel('x', fontsize=16)
    axs[0].set_ylabel('$C(x, 0)$', fontsize=16)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    # Configure Numerical Solution subplot
    axs[1].set_title('Numerical Solution', fontsize=18)
    axs[1].set_xlabel('x', fontsize=16)
    axs[1].set_ylabel('$C(x, 0)$', fontsize=16)
    axs[1].legend(fontsize=12)
    axs[1].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, 'Analytical_vs_Numerical.png')
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(f'Figure saved to {fig_path}')


if __name__ == "__main__":
    main()
