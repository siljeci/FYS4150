import numpy as np
import ising_model as ising
import matplotlib.pyplot as plt
import time
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('talk')

MC_cycles = 100000
num_spins = 20
Temp = [1.0, 2.4]
ordered = False

def plot_MC_cycles(Temp):
    """
    Plots the mean energy and the mean magnetization as functions of Monte
    Carlo cycles. This is being done for the two temperatures.

    Input:
    Temp: The temperatures (Two values in a list)
    """
    MCsteps = np.linspace(1, MC_cycles, MC_cycles, endpoint=True)

    for i, T in enumerate(Temp):
        # Random spin configuration:
        spin_matrix_R = np.random.choice((-1, 1), (num_spins, num_spins))

        time1 = time.time()
        exp_values_R = ising.MC(spin_matrix_R, MC_cycles, T)
        time2 = time.time()
        print("Montecarlo time used:", time2-time1)

        # Mean values:
        energy_avg_R = np.cumsum(exp_values_R[:,0])/np.arange(1, MC_cycles+1)
        magnet_abs_avg_R = np.cumsum(exp_values_R[:, 4])/np.arange(1, MC_cycles+1)

        # Mean values per spin:
        Energy_R = energy_avg_R/num_spins**2
        MagnetizationAbs_R = magnet_abs_avg_R/num_spins**2

        # Ordered spin confiuration:
        spin_matrix_O = np.ones((num_spins,num_spins), np.int8)

        time1 = time.time()
        exp_values_O = ising.MC(spin_matrix_O, MC_cycles, T)
        time2 = time.time()
        print("Montecarlo time used:", time2-time1)

        # Mean values:
        energy_avg_O = np.cumsum(exp_values_O[:,0])/np.arange(1, MC_cycles+1)
        magnet_abs_avg_O = np.cumsum(exp_values_O[:, 4])/np.arange(1, MC_cycles+1)

        # Mean values per spin:
        Energy_O = energy_avg_O/(num_spins**2)
        MagnetizationAbs_O = magnet_abs_avg_O/(num_spins**2)

        # Plot:
        fig, ax = plt.subplots(2, 1, figsize=(18, 10), sharex=True) # plot the calculated values
        plt.suptitle('{}'.format('Energy and magnetization vs. number of Monte Carlo cycles. T=%s'%str(T)))

        ax[0].plot(MCsteps, Energy_O, color='C0', label='Ordered')
        ax[0].plot(MCsteps, Energy_R, color='C1', label='Random')
        ax[0].set_ylabel("Mean energy", fontsize=15)
        ax[0].legend(loc='best')

        ax[1].plot(MCsteps, MagnetizationAbs_O, color='C0', label='Ordered')
        ax[1].plot(MCsteps, MagnetizationAbs_R, color='C1', label='Random')
        ax[1].set_ylabel("Mean magnetization (abs. value)", fontsize=15)
        ax[1].set_xlabel("Monte Carlo cycles", fontsize=15)
        ax[1].legend(loc='best')

    plt.show()

plot_MC_cycles(Temp)
