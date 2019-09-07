'''
Bifurcation diagram of the discrete system in mean field approximation
'''

import numpy as np
from math import fsum
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def MeanField(stateDensities, t, w0w2, k=1, C=5):
    # I think Chen uses k=3 ?

    # Computing the state dynamics rates
    w0 = w0w2*C
    w2 = C
    # Initialize the system of ODEs
    statesDerivative = []
    M = len(stateDensities)
    # For each density the corresponding ODE is generated depending on M
    for index, density in enumerate(stateDensities):
        otherDensities = stateDensities
        # A list is needed to be able to delete an entry, since arrays have a pre-specified length which may not change
        otherDensities = otherDensities.tolist()
        del otherDensities[index]
        # noise is the dynamics described by w0
        noise = w0 / (M-1) * sum(otherDensities) - w0 * density
        # stateDynamics are the triplet interactions. For each y_i a term is added in a for loop to make it work for different M
        stateDynamics = 0
        for otherDensity in otherDensities:
            stateDynamics += (density**2 * otherDensity - otherDensity**2 * density) * w2 * k**2
        # For each density the corresponding ODE is appended to the list
        statesDerivative.append(noise + stateDynamics)
    # Return the system of ODEs
    return statesDerivative


# Defining the initial conditions as follows: initialDensities = [X, Y, Z, .... A, B] depending on M
initialDensities = [0.8, 0.2]

tMax = 100
tSteps = 1000
# Choose the number of w0w2 ratio points for which the bifurcation diagram must be computed
N_w0w2 = 40
M = len(initialDensities)
t = np.linspace(0, tMax, tSteps)
# The following line makes sure that we have more points on the last part for a better looking diagram
w0w2 = np.hstack((np.linspace(0, 1/(M+1), int(N_w0w2/1.4)), np.linspace(1/(M+1), 1, N_w0w2)))
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

# Check if the densities add up to 1
if fsum(initialDensities) == 1:
    markers = ['x', '_', '|', 'o', '^', 's', 'p']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # For each w0w2 ratio we integrate our system and plot the steady state solutions, i.e. the last entry in the list
    for ratio in range(len(w0w2)):
        stateDensities = odeint(MeanField, initialDensities, t, args=(w0w2[ratio],))
        for var in range(M):
            ax.plot(w0w2[ratio], stateDensities[-1, var], marker=markers[var], markersize=3,
                    color=colors[var], label=labels[var] if ratio == 0 else "", alpha=1)
    # Drawing the analytical transition points as a thin line
    ax.plot([0.25, 0.25], [0, 1], 'k--', alpha=0.1)
    ax.plot([0, 1], [1/M, 1/M], 'k--', alpha=0.1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.xlabel(r'$w_0 / w_2$', fontsize=15)
    plt.legend(loc='upper right')

    ax.minorticks_on()
    plt.yticks(np.arange(0, 1.01, step=0.1), fontsize=13)
    plt.xticks(np.arange(0, 1.01, step=0.1), fontsize=13)
    #plt.savefig("FILENAME.pdf", bbox_inches="tight")
    plt.show()

else:
    print('ERROR: sum(initialDensities) != 1')

#
#
#
#
# def MeanField(x, t, w0w2, M):
#     C = 0.5
#     w0 = w0w2*C
#     w2 = C
#     k = 1
#     y = (1 - x) / (M-1)
#     dxdt = (y-x) * w0 + (x**2 * y - y**2 * x) * w2 * k**2 * (M-1)
#     return dxdt
#
#
#
# stateDensities = odeint(MeanFieldSum, initialDensities, t, args=(0.1,))
# # ax.plot(t, stateDensities)
# # plt.show()
#
#
# x0 = 0.3
# last = 500
# for i in range(N_w0w2):
#     x = odeint(MeanField, x0, t, args = (w0w2[i], M,))
#     # Plotting
#     ax.plot(np.ones(last) * w0w2[i], x[-last:], ',k')
#     ax.plot(np.ones(last) * w0w2[i], (1 - x[-last:]) / (M-1), ',r')
#
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# plt.show()
#
# Als je x=y=z kiest als beginconditie dan krijg je geen bifurcatiediagram omdat ze allemaal precies gelijk veranderen, voor elke andere willekeurige combinatie krijg je hetzelfde bifurcatiediagram,
# Daarnaast mis je in het Bifurcateidiagram het doorgetrokken stukje bij 0.33 die instabiel is en de andere instanbiele branches.
#
# if M == 2:
#     initialDensities = [0.4, 0.6]
#
#     for ratio in range(len(w0w2)):
#         stateDensities = odeint(MeanFieldSum, initialDensities, t, args=(w0w2[ratio],))
#         print(stateDensities)
#         ax.plot(w0w2[ratio], stateDensities[-1, 0], marker='x', markersize=5,
#                 color='tab:blue', label='[stateDensities]' if ratio == 0 else "", alpha=1)
#         ax.plot(w0w2[ratio], stateDensities[-1, 1], marker='o', markersize=5, mfc='none',
#                 color='tab:red', label='[Y]' if ratio == 0 else "", alpha=1)
#     ax.plot([.25, .25], [0, 1], 'k--', alpha=0.1)
#     ax.plot([0, 1], [.5, .5], 'k--', alpha=0.1)
#
#
# if M == 3:
#     initialDensities = [0.5, 0.3, 0.2]
#
#     for ratio in range(len(w0w2)):
#         stateDensities = odeint(MeanFieldSum, initialDensities, t, args=(w0w2[ratio],))
#         ax.plot(w0w2[ratio], stateDensities[-1, 0], marker='x', markersize=5,
#                 color='g', label='[stateDensities]' if ratio == 0 else "", alpha=1)
#         ax.plot(w0w2[ratio], stateDensities[-1, 1], marker='_', markersize=7,
#                 color='tab:red', label='[Y]' if ratio == 0 else "", alpha=1)
#         ax.plot(w0w2[ratio], stateDensities[-1, 2], marker='|', markersize=7,
#                 color='tab:blue', label='[Z]' if ratio == 0 else "", alpha=1)
#     ax.plot([0.25, 0.25], [0, 1], 'k--', alpha=0.1)
#     ax.plot([0, 1], [1/3, 1/3], 'k--', alpha=0.1)
#
# if M == 4:
#     initialDensities = [0.4, 0.3, 0.2, 0.1]
#
#     for ratio in range(len(w0w2)):
#         stateDensities = odeint(MeanFieldSum, initialDensities, t, args=(w0w2[ratio],))
#         ax.plot(w0w2[ratio], stateDensities[-1, 0], marker='x', markersize=5,
#                 color='g', label='[W]' if ratio == 0 else "", alpha=1)
#         ax.plot(w0w2[ratio], stateDensities[-1, 1], marker='_', markersize=7,
#                 color='tab:red', label='[stateDensities]' if ratio == 0 else "", alpha=1)
#         ax.plot(w0w2[ratio], stateDensities[-1, 2], marker='|', markersize=7,
#                 color='tab:blue', label='[Y]' if ratio == 0 else "", alpha=1)
#         ax.plot(w0w2[ratio], stateDensities[-1, 3], marker='o', markersize=5, mfc='none',
#                 color='y', label='[Z]' if ratio == 0 else "", alpha=1)
#     ax.plot([0.25, 0.25], [0, 1], 'k--', alpha=0.1)
#     ax.plot([0, 1], [1/4, 1/4], 'k--', alpha=0.1)
#
#     dXdt = [
#         w0 / (M-1) * (y + z) - w0 * x + w2 * k**2 * (x**2 * y - y**2 * x + x**2 * z - z**2 * x),
#         w0 / (M-1) * (z + x) - w0 * y + w2 * k**2 * (y**2 * z - z**2 * y + y**2 * x - x**2 * y),
#         w0 / (M-1) * (x + y) - w0 * z + w2 * k**2 * (z**2 * x - x**2 * z + z**2 * y - y**2 * z)
#     ]
#     return dXdt
