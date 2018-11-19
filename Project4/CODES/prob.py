import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import ising_model as ising
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('talk')

MC_cycles = [400000, 1000000]
num_spins = 20
Temp = [1.0, 2.4]
ordered = True

def plotProbability(ordered):
    """
    Function which plots the probabilities of finding an energy E. Being done
    for two temperatures in the list Temp, and for two corresponding numbers
    of Monte Carlo cycles.

    Input:
    ordered: True if ordered, False if random
    """
    for cycles, T in zip(MC_cycles, Temp):
        # Initializing the confiuration of the input spin matrix:
        if ordered == False:
            spin_matrix = np.random.choice((-1, 1), (num_spins, num_spins))
        else:
            spin_matrix = np.ones((num_spins,num_spins), np.int8)

        t1 = time.time()
        exp_values = ising.MC(spin_matrix, cycles, T)
        t2 = time.time()
        print("Montecarlo time used:", t2-t1)

        #If the temperature is 1.0:
        if T == 1.0:
            spacing = 30

            norm = 1/float(cycles-20000)
            energy_avg = np.sum(exp_values[19999:, 0])*norm
            energy2_avg = np.sum(exp_values[19999:, 2])*norm
            energy_var = (energy2_avg - energy_avg**2)/(num_spins**2)

            E = exp_values[19999:, 0]
            E = E/(num_spins**2)

        #If the temperature is 2.4:
        if T == 2.4:
            spacing = 160

            norm = 1/float(cycles-40000)
            energy_avg = np.sum(exp_values[39999:, 0])*norm
            energy2_avg = np.sum(exp_values[39999:, 2])*norm
            energy_var = (energy2_avg - energy_avg**2)/(num_spins**2)

            E = exp_values[39999:, 0]
            E = E/(num_spins**2)

        print("Variance T=%s:"%str(T), energy_var)

        n, bins, patches = plt.hist(E, spacing, facecolor='blue')
        plt.xlabel('$E$')
        plt.ylabel('Number of times the given energy appears')
        plt.title('Energy distribution at  $k_BT=%s$'%str(T))
        plt.grid(True)
        plt.show()

plotProbability(ordered)
