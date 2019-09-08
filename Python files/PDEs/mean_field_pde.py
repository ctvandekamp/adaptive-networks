'''
This scripts solves the continuous state adaptive network model in mean field approximation
This script is vectorised and uses hankel and toeplitz matrices
Data is output in a file
This script is compatible with the HPC cluster for large parameter sweeps
'''


import numpy as np
from scipy.linalg import hankel, toeplitz      # to create circular matrix
import matplotlib.pyplot as plt
import scipy.stats as stats  # for initialising als normal distribution
import os
import pickle
import sys


def averagingintegralfunc(states, x_steps):
    S = np.zeros(x_steps)
    for x in range(x_steps):
        for i in range(round(x_steps/4)):
            S[x] += states[(x+i) % x_steps]*states[(x-i) % x_steps]
    return S


# Variables
# Replace with float(sys.argv[i]) to be able to run parameter sweeps on the HPC cluster
n = 1               # float(sys.argv[1])
m = 18.6            # float(sys.argv[2])
meandegree = 1      # float(sys.argv[3])
# m = m/meandegree**2
eps = 10 ^ (-1)
t_steps = 300000
x_steps = 800
t_max = 1000


dt = t_max/t_steps
domega = 2*np.pi/x_steps

# Initialisation
states = np.zeros((t_steps, x_steps))
omega = np.arange(-np.pi, np.pi, domega)

IC = """
Gaussian_initial_distr = True
if Gaussian_initial_distr:
    # Start with Gaussian distribution
    states[0, :] = stats.norm.pdf(omega, 0, .1)  # + stats.norm.pdf(omega, -1.5, 0.87)
else:
    # Start with uniform distribution
    states[0, :] = 1/(2*np.pi)
    """

exec(IC)

A = domega*np.sum(states[0, :])
print('Initial area: ', A)
states[0, :] = states[0, :] / A     # Normalise distribution

# Time simulation
if 1:
    for t in range(t_steps-1):
        if t % 100 == 0:
            print('Area at t =', t, ' equals ', np.sum(states[t, :])*domega)
            print("calculating for t =", t)

        # Explanation of the discrete averaging integral underneath.

        H2 = np.append(0, states[t, :round(3*x_steps/(4))-1:-1])
        H = toeplitz(states[t, :], H2)
        H = H[:, 1:]

        G2 = np.append(states[t, -1], states[t, 0:round(x_steps/(4))])
        G = hankel(states[t, :], G2)
        G = G[:, 1:]

        averagingintegral = np.sum(H*G, axis=1)
        # averagingintegral = averagingintegralfunc(states[t, :], x_steps)
        # if (abs(averagingintegral-averagingintegral_check) > eps).any():
        #     print('ERROR: averaging integral')
        #     print(averagingintegral)
        #     print(averagingintegral_check)
        #     print(averagingintegral.shape)
        #     print(averagingintegral_check.shape)
        #     break
        # print(averagingintegral*domega)
        # print(states[t, :])
        # Discrete version of integral over complete state set of the averaging integral
        totalaverages = sum(averagingintegral)
        states[t+1, :] = dt*n/2/np.pi + states[t, :] * \
            (1 - dt*n) + dt*domega*m * meandegree**2 * averagingintegral[:] - \
            dt*m*meandegree**2 * states[t, :] * domega**2 * totalaverages

if True:
    # Reducing array sizes
    states = states[slice(0, t_steps+1, 100), :]

    # Saving
    data = {'system': {'n': n, 'm': m, 'meandegree': meandegree},
            'steps': {'t_steps': t_steps, 'x_steps': x_steps, 't_max': t_max},
            'ic': IC,
            'states_t%10000': states}

    filenumber = 0
    while os.path.exists("data{}_{}_{}_{}.dat".format(str(filenumber), str(n), str(m), str(meandegree))):
        filenumber += 1
    file = open("data{}_{}_{}_{}.dat".format(str(filenumber),
                                             str(n), str(m), str(meandegree)), "wb")
    pickle.dump(data, file)
    file.close()
    print('data saved succesfully to ', file)


if False:
    fig, ax = plt.subplots()
    for t in np.arange(0, t_steps, 20):
        ax.plot(omega, states[t, :])
        #print(np.sum(states[t, :])*domega)
    plt.show()

###################################################

# the following lines create two matrices which kan be used for the discrete averaging integral (over k). For each x value
# (M matrix column) we need to multiply N/4 values of x-1 * x+1, x-2*x+2 etcetera including x*x (latter not needed). \
# we do this by creating two matrices, a toeplitz and a hankel matrix. The toeplitz matrix contains N/4 all
# elements on the left side of x, each x value has its own row. The hankel matrix has all N/4 values on the right side of x.
# Elementwise multiplication and summing along the rows should give the right discretisation for the averaging integral.
#
# N = 20
# M = np.reshape(np.arange(N**2), (N, N))         # creating testmatrix
#
# h1 = M[0, :]        # each x value has its own row in the toeplitz matrix
# # the elements on the top row of the toeplitz matrix. These contain x itself (not needed, then the area goes to infinity),
# # and then all N/4 elements to the left of x, such that we need to count back using the '-1'
# h2 = np.append(0, M[0, :round(3*N/4)-1:-1])
# h = toeplitz(h1, h2)
#
# g1 = M[0, :]        # each x value has its own row in the hankel matrix
# # the elements on the bottom row of the hankel matrix. Tese contain x itself (not needed) and alle N/4 elements to the
# # right. Since the bottom row is for X=N we start again by the zeroth element as the first element on the right of x.
# g2 = np.append(M[0, -1], M[0, 0:round(N/4)])
# g = hankel(g1, g2)
#
# print(h)
# print(g)
#
# # summing the values from pairwise multiplication.times domega gives the Riemann integral discretisation
# s = np.sum(g*h, axis=1)


# N = 20
# M = np.reshape(np.arange(N**2), (N, N))         # creating testmatrix
#
# h1 = M[0, :]        # each x value has its own row in the toeplitz matrix
# # the elements on the top row of the toeplitz matrix. These contain x itself (not needed, then the area goes to infinity),
# # and then all N/4 elements to the left of x, such that we need to count back using the '-1'
# h2 = np.append(0, M[0, :round(3*N/4)-1:-1])
# h = toeplitz(h1, h2)
#
# g1 = M[0, :]        # each x value has its own row in the hankel matrix
# # the elements on the bottom row of the hankel matrix. Tese contain x itself (not needed) and alle N/4 elements to the
# # right. Since the bottom row is for X=N we start again by the zeroth element as the first element on the right of x.
# g2 = np.append(M[0, -1], M[0, 0:round(N/4)])
# g = hankel(g1, g2)
#
# print(h)
# print(g)
#
# # summing the values from pairwise multiplication.times domega gives the Riemann integral discretisation
# s = np.sum(g*h, axis=1)
