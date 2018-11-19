"""
This program calculates and plot four thermodynamic quantities derived by the
Ising model using the MC functions from ising_model.py. The result is an ensemble
plot with results from the calculations of four different lattice sizes. The
plots show the thermodynamic quantities as functions of temperature with a
temperature step of 0.01 [kT/J]. The number of Monte Carlo cycles is set in
MC_cycles and the progress is printed to terminal. Finally, a nuerical approximation
to the critical temperature is calculated and printed to terminal.
"""

import numpy as np
import ising_model as ising
import matplotlib.pyplot as plt
import time
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('talk')

# Initial variables to loop over:
L = [40,60,80,100]
T = np.arange(2.1, 2.51, 0.01)
# Number of Monte Carlo cycles for the calculation:
MC_cycles = 200000
norm = 1.0/float(MC_cycles-40000)
# Initializing output matrices:
E = np.zeros((len(L),len(T)))
M_abs = np.zeros((len(L),len(T)))
m_av = np.zeros((len(L),len(T)))
C_v = np.zeros((len(L),len(T)))
X = np.zeros((len(L),len(T)))

# Looping over all lattice sizes and all temperature steps:
for i, spins in enumerate(L):
    spin_matrix = np.ones((L[i],L[i]), np.int8)

    for j, temp in enumerate(T):
        exp_values = ising.MC(spin_matrix, MC_cycles, temp)

        energy_avg = np.sum(exp_values[40000:, 0])*norm
        magnet_abs_avg = np.sum(exp_values[40000:, 4])*norm
        energy2_avg = np.sum(exp_values[40000:, 2])*norm
        magnet2_avg = np.sum(exp_values[40000:, 3])*norm
        energy_var = (energy2_avg - energy_avg**2)/((spins)**2)
        magnet_var = (magnet2_avg - magnet_abs_avg**2)/((spins)**2)

        E[i,j] = (energy_avg)/(spins**2)
        M_abs[i,j] = (magnet_abs_avg)/(spins**2)
        C_v[i, j] = energy_var/temp**2
        X[i, j] = magnet_var/temp
        print(temp)
    print(spins)


# Plotting the energy, magnetization, specific heat and magnetization:
fig, ax = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
for i in range(4):
    ax[0, 0].plot(T, E[i, :], label='L = {}'.format(L[i]))
    ax[0, 0].set_ylabel('Mean Energy')
    ax[0, 0].legend(loc='upper left', frameon=False)

    ax[0, 1].plot(T, M_abs[i, :], label='L = {}'.format(L[i]))
    ax[0, 1].set_ylabel('Mean abs. magnetization')
    ax[0, 1].legend(loc='upper right', frameon=False)

    ax[1, 0].plot(T, C_v[i, :], label='L = {}'.format(L[i]))
    ax[1, 0].set_ylabel('Specitfic Heat')
    ax[1, 0].legend(loc='upper left', frameon=False)

    ax[1, 1].plot(T, X[i, :], label='L = {}'.format(L[i]))
    ax[1, 1].set_ylabel('Susceptibility')
    ax[1, 1].legend(loc='upper right', frameon=False)

    ax[1, 0].set_xlabel('Temperature')
    ax[1, 1].set_xlabel('Temperature')

plt.suptitle('Numerical studies of phase transitions')
plt.savefig('PT_longrun.png')
plt.show()

# Calculating the critical temperature for an infinite lattice:
max_2 = np.argmax(X[2, :])
max_3 = np.argmax(X[3, :])
Tc_L2 = T[max_2]
Tc_L3 = T[max_3]

# exponent = -(1./v) with v = 1:
exponent = -1.0
Tc_infinity = Tc_L2 - ((Tc_L2-Tc_L3)/(L[2]**exponent-L[3]**exponent))*(L[2]**exponent)
print("T_C=",Tc_infinity)
