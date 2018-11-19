import numpy as np
import ising_model as ising
import matplotlib.pyplot as plt
import time
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('talk')

MC_cycles = 1000000
num_spins = 20
Temp = [1.0, 2.4]
ordered = False

def Accepted(ordered):
    """
    Function which plots the number of accepted configurations in the Metropolis
    algorithm for two temperatures and for an ordered- and a random initial
    configuration.

    Input:
    ordered: True if ordered, False if random
    """
    titles = ['Random input configuration', 'Ordered input configuration']
    fig, ax = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    MCsteps = np.linspace(0, MC_cycles, MC_cycles, endpoint=True)
    for i in range(2):
        # Initializing the confiuration of the input spin matrix:
        if ordered == False:
            spin_matrix = np.random.choice((-1, 1), (num_spins, num_spins))
        else:
            spin_matrix = np.ones((num_spins,num_spins), np.int8)

        t1 = time.time()
        exp_values_T1 = ising.MC(spin_matrix, MC_cycles, Temp[0])
        t2 = time.time()
        print("Montecarlo time used:", t2-t1)

        t1 = time.time()
        exp_values_T24 = ising.MC(spin_matrix, MC_cycles, Temp[1])
        t2 = time.time()
        print("Montecarlo time used:", t2-t1)

        # Extracting the number of accepted configs:
        accepted_list_T1 = exp_values_T1[:,5]
        accepted_list_T24 = exp_values_T24[:,5]

        # Changing to ordered input configuration:
        ordered = True

        # Plot:
        ax[i].set_title('{}'.format(titles[i]))
        ax[i].plot(MCsteps, accepted_list_T1, color='C2', label='T = 1.0')
        ax[i].plot(MCsteps, accepted_list_T24, color='C4', label='T = 2.4')
        ax[i].legend(loc='best')
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
    ax[1].set_xlabel("Monte Carlo cycles", fontsize=20)
    fig.text(0.04, 0.5, 'Number of accepted configurations', va='center', rotation='vertical')
    plt.show()

Accepted(ordered)
