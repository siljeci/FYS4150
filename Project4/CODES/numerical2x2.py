"""
Program which prints the numerical and the analytical solutions of a 2x2 lattice
to the terminal.
"""
import time
import numpy as np
import ising_model as ising

#============================Numerical solutions=============================
def numerical_solution(spin_matrix, num_cycles):
    exp_values = ising.MC(spin_matrix, num_cycles, temp)
    norm = 1.0/float(num_cycles)

    # <E>, <M>, <E^2>, <M^2> and <|M|>:
    energy_avg = np.sum(exp_values[:,0])*norm
    magnet_avg = np.sum(exp_values[:,1])*norm
    energy2_avg = np.sum(exp_values[:,2])*norm
    magnet2_avg = np.sum(exp_values[:,3])*norm
    magnet_abs_avg = np.sum(exp_values[:,4])*norm

    # Variances per spin:
    energy_var = (energy2_avg - energy_avg**2)/(num_spins**2)
    magnet_var = (magnet2_avg - magnet_avg**2)/(num_spins**2)

    # Expectation values and quantities per spin:
    Energy = energy_avg/(num_spins**2)
    Magnetization = magnet_avg/(num_spins**2)
    MagnetizationAbs = magnet_abs_avg/(num_spins**2)
    SpecificHeat = energy_var/(temp**2)
    Susceptibility = magnet_var/(temp)

    return Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility

#======================Initial conditions=====================================
max_cycles = 1000000
num_spins = 2
temp = 1.0
runs = np.logspace(2, int(np.log10(max_cycles)), (int(np.log10(max_cycles))-1), endpoint=True)
spin_matrix = np.ones((num_spins, num_spins), np.int8)

#=========Print to terminal for each magnitude of Monte Carlo cycles==========
for i, num_cycles in enumerate(runs):
    Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility = \
numerical_solution(spin_matrix, num_cycles)
    print('Monte Carlo cycles:{}'.format(num_cycles))
    print('Mean energy:{}'.format(Energy))
    print('Specific Heat:{}'.format(SpecificHeat))
    print('Mean Magenetization:{}'.format(Magnetization))
    print('Susceptibility:{}'.format(Susceptibility))
    print('Mean Absolute Magnetization:{}'.format(MagnetizationAbs))
    print('\n')

#=================Calculating the analytic solutions==========================
A_Energy = -(8.0*np.sinh(8.0))/(3.0 + np.cosh(8.0))/4.0
A_SpecificHeat = ((64.0/(3.0 + np.cosh(8.0)))*(np.cosh(8.0)-(np.sinh(8.0)**2)/\
(3.0 + np.cosh(8.0))))/4.0
A_Magnetization = 0.0
A_MagnetizationAbs = ((2.0*np.exp(8.0) +  4.0)/(3.0 + np.cosh(8.0)))/4.0
A_Susceptibility = (8.0*(np.exp(8.0)+1)/(3.0+np.cosh(8.0)))/4.0

#===============Print analytic solutions to terminal==========================
print('Analytic mean energy:{}'.format(A_Energy))
print('Analytic Specific heat:{}'.format(A_SpecificHeat))
print('Analytic mean Magenetization:{}'.format(A_Magnetization))
print('Analytic Susceptibility:{}'.format(A_Susceptibility))
print('Analytic mean absolute Magnetization:{}'.format(A_MagnetizationAbs))
print('\n')
