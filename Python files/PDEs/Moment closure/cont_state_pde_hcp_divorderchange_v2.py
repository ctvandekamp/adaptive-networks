'''
This script computes the time evolution of the continous state adaptive network model with moment closure approximation
This script is compatible with the HPC cluster (use sys.argv[i] for parameter sweeps)
If run on the cluster, one can use 128 bit floats (long doubles) for higher precision. These need to be converted back to 64 bit floats
to be compatible with a Windows machine.
Unfortunately, it is not vectorised and uses lots of for-loops to do the integration
If the time steps are not small enough the computation crashes. The precise reason for this is not clear.
'''


import numpy as np
import time
import pickle
import os
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats  # for initialising als normal distribution
#from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D


# Variables
# Using the sys library the variables can be read from a .txt file. This is very helpful if the code runs from the command line or on a hpc cluster.
eta = float(sys.argv[1])
sigma = float(sys.argv[2])
alpha = float(sys.argv[3])
beta = float(sys.argv[4])
# If the code runs in an ide, declare the system parameters in the normal fashion and uncomment the statements above this line.
#eta   = .1
#sigma = 1
#alpha = .1
#beta  = .1

# Actually coupled to the rates (that is high rates need lots of time steps and vice versa)
t_steps = 20000
omega_steps = 24             # Best if divisible by 4 to prevent rounding errors in summation boundary
t_max = 50                   # For tmax = 50 probably all simulations either converge completely or diverge to infinity
threshold = 1e-15            # If the pc cannot use 128 bit floats turn this value down


# Initialisation
# np.longdouble uses the highest precision floating numbers available on the system. For a windows pc these are 64 bit floats, on a linux machine (cluster) these are 128 bit floats
states = np.zeros((t_steps, omega_steps), dtype=np.longdouble)
links = np.zeros((t_steps, omega_steps, omega_steps), dtype=np.longdouble)
omega = np.linspace(-np.pi, np.pi, num=omega_steps, endpoint=True, dtype=np.longdouble)

dt = t_max/t_steps
domega = 2*np.pi/omega_steps


# Initial condition
# Little bit unorthodox but it works
# The ic is packed as a str such that it can be pickled and saved in a .dat file.
# This way one can always find the IC of a certain run
# exec(IC) executes the code inside the string object
IC = """
COS_initial_distr = True
if COS_initial_distr:
    states[0, :] = 1/(2*np.pi) + 0.1 *np.cos(omega)   #   stats.norm.pdf(omega, 0, 5)   #   #stats.norm.pdf(omega, 0, 1)   #+ stats.norm.pdf(omega, -1.5, 0.87)
#    links[0, :, :] = 1/(2*np.pi)
    MeshX, MeshY = np.meshgrid(omega,omega)
#    links[0, :, :] = bivariate_normal(MeshX,MeshY,1,1,0,0)
    links[0, :, :] = stats.multivariate_normal.pdf(np.dstack((MeshX, MeshY)), mean=[0,0], cov=[[1,0],[0,1]])

else:
    # Start with uniform distribution
    states[0, :] = 1/(2*np.pi)
    links[0, :, :] = 1/(2*np.pi)  # Gaussian
    """

exec(IC)


A = domega * np.sum(states[0, :])
# Normalise distribution on interval [-pi, pi] (other wise IC is infeasible)
states[0, :] = states[0, :] / A

print('initial area is ', A)

# Some more initialisation. The convergence parameters allow for breaking the loop when convergence is achieved
looptime = 0
convergenceth = 1e-6
convergencecheck = 0
convergenceachieved = False
area_infeasible = 0

# Time simulation
if True:
    for t in range(t_steps-1):
        startingtime = time.time()

        Area = np.sum(states[t, :])*domega

        # Once every n time steps some feasibility and convergence conditions are checked and area + looptime are printed
        if t % 40 == 0:
            print('area at t =', t, ' equals ', Area)
            print('loop took ', looptime, ' seconds')
            if abs(Area) > 10 or np.isnan(Area):
                print('area infeasible')
                area_infeasible = t
                break
            print("calculating for t =", t)

            if (states[t, :] - states[t-1, :]).all() < convergenceth and (links[t, :, :] - links[t-1, :, :]).all() < convergenceth and t > 1:
                convergencecheck += 1
                if convergencecheck == 20:
                    # If for 20 * 40  consecutive time steps the distribution change was smaller than convergenceth then we say convergence is achieved
                    # this detection does not work very well
                    convergenceachieved = True
                    print('CONVERGENCE ACHIEVED')
                    break

        # The next for loop sets all values smaller than threshold to zero to minimise the numerical errors caused by small numbers/ too little precision
        # it also lowers the number of runs that crash...
        for p in range(omega_steps):
            if states[t, p] < threshold:
                states[t, p] = 0
                print('value ', p, ' set to 0')
            for q in range(omega_steps):
                if links[t, p, q] < threshold:
                    links[t, p, q] = 0

        # statematrix is a matrix [ [p1q1, p1q2]; [p2q1, p2q2]; [p3q1, p3q2]]
        statematrixhelp = np.tile(states[t, :], (omega_steps, 1))
        statematrix = statematrixhelp.T * statematrixhelp

        # Initialisation of integral arrays
        # Again data type long double is used
        Riemannstate = np.zeros((omega_steps), dtype=np.longdouble)
        Riemannlinks1 = np.zeros((omega_steps, omega_steps), dtype=np.longdouble)
        Riemannlinks2 = np.zeros((omega_steps, omega_steps), dtype=np.longdouble)

        # Summation in pde for f(x;t)
        for p in range(omega_steps):
            for j in range(omega_steps):          # -1 already captured by range()-command
                for i in range(round(omega_steps/4)):
                    RS1 = 0
                    RS2 = 0
                    if states[t, j] != 0:
                        # The modulo % operator has the same effect as creating an even extension of the distribution on the boundaries.
                        # We need this to prevent the integral boundaries going out of the state set
                        RS1 = links[t, p-i, j] / states[t, j] * links[t, j, (p+i) % omega_steps]
                    if states[t, p] != 0:
                        RS2 = - links[t, j-i, p] / states[t, p] * links[t, p, (j+i) % omega_steps]

                    Riemannstate[p] += RS1 + RS2

        # First summation in pde for l(x,y;t)
            for q in range(omega_steps):
                for j in range(omega_steps):
                    RL1 = eta/(2*np.pi) * (links[t, p, j] + links[t, j, q])
                    RL2 = 0
                    RL3 = 0
                    RL4 = 0
                    RL5 = 0
                    if states[t, p] != 0:
                        RL2 = - sigma * links[t, q, p] / states[t, p] * links[t, p, j]
                    if states[t, q] != 0:
                        RL3 = - sigma * links[t, p, q] / states[t, q] * links[t, q, j]
                    if states[t, j] != 0:
                        RL4 = sigma * links[t, q, j] / states[t, j] * \
                            links[t, j, (-q+2*p) % omega_steps]
                        RL5 = sigma * links[t, p, j] / states[t, j] * \
                            links[t, j, (-p+2*q) % omega_steps]

                    Riemannlinks1[p, q] += RL1 + RL2 + RL3 + RL4 + RL5

                    # Second summation in pde for l(x,y;t)
                    for i in range(round(omega_steps/4)):
                        RL6 = 0
                        RL7 = 0
                        RL8 = 0
                        RL9 = 0
                        if states[t, j] != 0:
                            RL6 = links[t, q, j] / states[t, j] * links[t, j, p-i] / \
                                states[t, j] * links[t, j, (p+i) % omega_steps]
                            RL7 = links[t, p, j] / states[t, j] * links[t, j, q-i] / \
                                states[t, j] * links[t, j, (q+i) % omega_steps]
                        if states[t, p] != 0:
                            RL8 = - links[t, q, p] / states[t, p] * links[t, p, j-i] / \
                                states[t, p] * links[t, p, (j+i) % omega_steps]
                        if states[t, q] != 0:
                            RL9 = - links[t, p, q] / states[t, q] * links[t, q, j-i] / \
                                states[t, q] * links[t, q, (j+i) % omega_steps]

                        Riemannlinks2[p, q] += RL6 + RL7 + RL8 + RL9

        states[t+1, :] = states[t, :] + dt * (eta * (1/(2*np.pi) - states[t, :]) +
                                              sigma * domega**2 * Riemannstate[:])

        links[t+1, :, :] = links[t, :, :] + dt * (alpha * statematrix[:, :] - (2*eta + beta) * links[t, :, :] +
                                                  domega * Riemannlinks1[:, :] + sigma * domega**2 * Riemannlinks2[:, :])

        # Compute the computation time of the current time step
        looptime = time.time() - startingtime


# Save all results and conditions using pickle

if True:
    # Saving in float64 data type since 64 bit systems cannot handle float128, while the hpc cluster can
    states = states.astype('float64')
    links = links.astype('float64')

    # Reducing array sizes for saving purposes, not needed if <5000 time steps
    states = states[slice(0, t+1, 10), :]
    links = links[slice(0, t+1, 10), :, :]

    # Saving
    # data contains all information; from the simulation results to all inital conditions
    data = {'system': {'eta': eta, 'sigma': sigma, 'alpha': alpha, 'beta': beta},
            'steps': {'t_steps': t_steps, 'omega_steps': omega_steps, 't_max': t_max},
            'ic': IC,
            'threshold': threshold,
            'states_t%10': states, 'links_t%10': links,
            'convergence': convergenceachieved, 'infeasible': area_infeasible}

    # Determining file name: name contains relevant system parameters and an extra number if the name already exists
    # making sure no file is overwritten
    filenumber = 0
    while os.path.exists("data_IC6_{}_{}_{}_{}_{}.dat".format(str(filenumber), str(eta), str(sigma), str(alpha), str(beta))):
        filenumber += 1
    file = open("data_IC6_{}_{}_{}_{}_{}.dat".format(str(filenumber),
                                                     str(eta), str(sigma), str(alpha), str(beta)), "wb")
    pickle.dump(data, file)
    file.close()
    print('data saved succesfully to ', file)


# Plotting

if False:
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    MeshX, MeshY = np.meshgrid(omega, omega)
    for t in np.arange(0, t_steps-1, 1):
        ax1.plot(omega, states[t, :])
    for t in np.arange(0, t_steps-1, 1):
        ax2.plot_surface(MeshX, MeshY, links[t, :, :], alpha=0.5)

    plt.show()
    #plt.savefig("plot{}_{}_{}_{}_{}.png".format(str(filenumber), str(eta), str(sigma), str(alpha), str(beta)))
