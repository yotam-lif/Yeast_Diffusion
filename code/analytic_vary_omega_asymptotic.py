import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

# Define parameters
resolution_density = 2 * 10 ** -2
r = 0.1
N_max = 10 ** 5
L_x = 10 ** 3  # Fixed L_x value

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


# Iterate over values for Omega
omega_values = [0.1, 0.3, 0.5, 0.7, 1.0]
plots_data = []
num_x = int(L_x / resolution_density)
X = np.linspace(-L_x, L_x, num_x)

for omega in omega_values:

    # Compute the solution for infinite z
    C_infinite = C_infinite_z(X, L_x, r, omega, N_max)
    plots_data.append((omega, C_infinite))

plt.style.use(['science', 'no-latex'])
plt.figure(figsize=(14, 10))
mask = (X >= 100) & (X <= 200)
relx = np.log10(X[mask])

# x >> L_D
for omega, data in plots_data:
    # Asymptotically y(x) = (pi * omega * x) ^ -1
    rely = np.log10(data[mask])
    rely += np.log10(np.pi)
    rely += np.log10(omega)
    rely *= -1
    plt.plot(relx, rely, label=f'$\\Omega = {omega}$', linewidth=2)

# Add a y=x line (45-degree line in log-log plot)
plt.plot(relx, relx, 'k--', label='$y=x$', linewidth=1.5)

plt.title(f'Log-Log Plot: $L_x = {L_x}$; $r={r}$; $N_{{\\text{{max}}}}$ = {N_max}', fontsize=18)
plt.xlabel('$log(x)$', fontsize=18)
plt.ylabel('$log(-C(x,0) / \\frac{r}{\\Omega}) + log(\\pi \\Omega)$', fontsize=18)
plt.legend(fontsize=16)
plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(figs_dir, f'vary_omega_asymptotic_large.png'), dpi=200)
plt.close()

# x << L_D
plt.style.use(['science', 'no-latex'])
plt.figure(figsize=(14, 10))
mask = (X >= 0) & (X <= 0.01)
relx = np.log10(X[mask])

for omega, data in plots_data:
    # Asymptotically y(x) = (pi * omega * x) ^ -1
    rely = np.log10(data[mask])
    rely += np.log10(np.pi)
    rely += np.log10(omega)
    rely *= -1
    plt.plot(relx, rely, label=f'$\\Omega = {omega}$', linewidth=2)

# Add a y=x line (45-degree line in log-log plot)
plt.plot(relx, relx, 'k--', label='$y=x$', linewidth=1.5)

plt.title(f'Log-Log Plot: $L_x = {L_x}$; $r={r}$; $N_{{\\text{{max}}}}$ = {N_max}', fontsize=18)
plt.xlabel('$log(x)$', fontsize=18)
plt.ylabel('$log(-C(x,0) / \\frac{r}{\\Omega}) + log(\\pi \\Omega)$', fontsize=18)
plt.legend(fontsize=16)
plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(figs_dir, f'vary_omega_asymptotic_large.png'), dpi=200)
plt.close()
