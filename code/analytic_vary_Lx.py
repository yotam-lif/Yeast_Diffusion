import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

# Define parameters
resolution_density = 4 * 10 ** -2
r = 0.1
N_max = 10 ** 5
omega = 0.2  # Fixed omega value

# Create directory for saving figures
figs_dir = '../figs'
os.makedirs(figs_dir, exist_ok=True)


def k(n, L_x):
    return np.pi * n / (2 * L_x)


def Theta(x):
    return np.where(x < 0, 0, 1)


# Define the solution C(x) for infinite z
def C_infinite_z(_x, _L_x, _r, _omega, _N_max):
    sum_result = np.zeros_like(_x, dtype=float)
    for n in range(_N_max + 1):
        kodd = k(2 * n + 1, _L_x)
        numerator = np.sin(kodd * _x)
        denominator = kodd + _omega
        sum_result += numerator / denominator
    return Theta(-_x) + (sum_result / _L_x)


# Iterate over values for L_x
Lx_values = [10, 20, 50, 100]
plots_data = []

for L_x in Lx_values:
    num_x = int(L_x / resolution_density)

    # Create the grid
    X = np.linspace(-L_x, L_x, num_x)

    # Compute the solution for infinite z
    C_infinite = C_infinite_z(X, L_x, r, omega, N_max)
    plots_data.append((L_x, C_infinite))

# Plot all 1D plots for infinite z together
plt.style.use(['science', 'no-latex'])
plt.figure(figsize=(14, 10))
for L_x, data in plots_data:
    x = np.linspace(-L_x, L_x, len(data))
    plt.plot(x, data, label=f'$L_x = {L_x}$', linewidth=2)

plt.axhline(y=0.5, linestyle='--', color='grey', linewidth=1.5)
plt.text(Lx_values[-1], 0.51, '0.5', verticalalignment='bottom', horizontalalignment='right', fontsize=16)

plt.axhline(y=1.0, linestyle='--', color='grey', linewidth=1.5)
plt.text(Lx_values[-1], 1.01, '1', verticalalignment='bottom', horizontalalignment='right', fontsize=16)

plt.axvline(x=1 / omega, linestyle='--', color='black', linewidth=1.5)
plt.text(1 / omega, 0.6, f'$+1/\\Omega = {1 / omega:.2f}$', verticalalignment='bottom', horizontalalignment='left', fontsize=16, rotation=90)

plt.axvline(x=-1 / omega, linestyle='--', color='black', linewidth=1.5)
plt.text(-1 / omega, 0.3, f'$-1/\\Omega = {-1 / omega:.2f}$', verticalalignment='bottom', horizontalalignment='right', fontsize=16, rotation=90)

plt.title(f'$\\Omega = {omega}$; $r={r}$; $N_{{\\text{{max}}}}$ = {N_max}', fontsize=18)
plt.xlabel('x', fontsize=18)
plt.ylabel('$C(x,0) / \\frac{r}{\\Omega}$', fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(figs_dir, f'vary_Lx_plots_Omega_{omega}.png'), dpi=200)
plt.close()
