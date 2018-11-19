import numpy as np
import numba

@numba.njit(cache = True)
def initial_energy(spin_matrix, num_spins):
    """
    Calculating the initial energy and the initial magnetization for the
    spin matrix

    Input:
    spin_matrix: The initial lattice (ordered or random) as a matrix
    num_cycles: Number of Monte Carlo cycles

    Output:
    E: Initial energy
    M: Initial magnetization
    """
    E = 0
    M = spin_matrix.sum()

    for i in range(num_spins):
        for j in range(num_spins):
            left = spin_matrix[i-1, j] if i > 0 else spin_matrix[num_spins-1, j]
            above = spin_matrix[i, j-1] if j > 0 else spin_matrix[i, num_spins-1]

            E -= spin_matrix[i,j]*(left + above)

    return E, M


@numba.njit(cache=True)
def MC(spin_matrix, num_cycles, temperature):
    """
    The main program which uses the Metropolis algorithm for calculating
    expectation values for a lattice of a size.

    Input:
    spin_matrix: The initial lattice (ordered or random) as a matrix
    num_cycles: Number of Monte Carlo cycles
    temperature: Temperature

    Output:
    exp_values: A matrix with the energy, magnetization, energy**2, magnetization**2,
    absolute value of the magnetization and the counter for finding the number of
    accepted configurations
    """

    num_spins = len(spin_matrix)
    exp_values = np.zeros((int(num_cycles), 6))
    E, M = initial_energy(spin_matrix, num_spins)

    # Precalculate the probabilities for the energy difference:
    w = np.zeros(17, np.float64)
    for delta_energy in range(-8, 9, 4):
        w[delta_energy+8] = np.exp(-delta_energy/temperature)

    accepted = 0

    # Start the Metropolis algorithm:
    for i in range(1, num_cycles+1):
        for j in range(num_spins*num_spins):
            ix = np.random.randint(num_spins)
            iy = np.random.randint(num_spins)

            left = spin_matrix[ix - 1, iy] if ix > 0 else spin_matrix[num_spins - 1, iy]
            right = spin_matrix[ix + 1, iy] if ix < (num_spins - 1) else spin_matrix[0, iy]

            above = spin_matrix[ix, iy - 1] if iy > 0 else spin_matrix[ix, num_spins - 1]
            below = spin_matrix[ix, iy + 1] if iy < (num_spins - 1) else spin_matrix[ix, 0]

            delta_energy = (2 * spin_matrix[ix, iy] * (left + right + above + below))

            if np.random.random() <= w[delta_energy+8]:
                spin_matrix[ix, iy] *= -1.0
                E += delta_energy
                M += 2*spin_matrix[ix, iy]
                accepted += 1

        exp_values[i-1,0] += E
        exp_values[i-1,1] += M
        exp_values[i-1,2] += E**2
        exp_values[i-1,3] += M**2
        exp_values[i-1,4] += np.abs(M)
        exp_values[i-1,5] += accepted

    return exp_values
