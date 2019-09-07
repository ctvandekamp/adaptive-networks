'''
Computing the bifurcation diagram of the 2 or 3 state discrete adaptive network with pair level closure approximation
'pairlevel_closure.py' needs to be imported for functions
'''


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import isclose, fsum
# Import our dynamics ODE's
from pairlevel_closure import Dynamics2, Dynamics3
from saveloadarray import SaveArray

# Definition of the initial conditions
# If M = 2: moments = [X Y XX XY YY]
# If M = 3: moments = [X, Y, Z, XX, XY, YY, YZ, ZZ, ZX]

moments = [0.6, 0.4, 0.3, 0.5, 0.2]
# moments = [0.4, 0.35, 0.25, 0.1, 0.2, .15, .05, .3, .2]
denN = len(moments)
# Check if M = 2 or M = 3
if denN == 5 or denN == 9:
    # Definition of the initial conditions
    Nw0 = 40
    w0Max = 1
    tMax = 300
    tSteps = 30
    w2 = 1        # Chen uses 0.2
    a = 0.5         # Chen uses 0.5
    d = 0.2         # Chen uses 0.1
    doSaveSol = 0
    filename = "solution_tstep2.dat"

    # Define the range of w0 rates for which to compute the bifurcation
    w0 = np.linspace(0, w0Max, Nw0)
    #
    # The next line makes a higher resolution in the critical area
    # w0 = np.hstack((np.linspace(0, 0.5, Nw0), np.linspace(0.5, 0.6, Nw0), np.linspace(0.6, 1, Nw0)))
    t = np.linspace(0, tMax, tSteps)

    # Solution in a 3D matrix, first dim: w0, second dim: t, third dim: all moments, X,Y,XY etc.
    sol = np.zeros((Nw0, tSteps, denN))

    if denN == 5:
        # Check connectivity of the network, so there are no loose nodes
        if fsum(moments[0:2]) == 1 and fsum(moments[2:]) == 1:
            # Integrate the system for all different noise rates in w0
            for noise in range(len(w0)):
                sol[noise, :, :] = odeint(Dynamics2, moments, t, args=(w0[noise], w2, a, d))
                print(str(noise) + " / " + str(len(w0)))

            # Saving our data
            if doSaveSol:
                SaveArray(sol, filename)
        else:
            print('ERROR: densities do no add up / network not connected')

# Plot only the last value corresponding to each noise rate
fig, (ax1, ax2) = plt.subplots(1, 2)  # , figsize=(10, 7.5))

ax1.plot(w0, sol[:, -1, 0], 'k.', markersize=8, alpha=0.3, label=r'$[X]$')
ax1.plot(w0, sol[:, -1, 1], 'kx', markersize=5, alpha=0.5, label=r'$[Y]$')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.minorticks_on()

ax2.plot(w0, sol[:, -1, 2], 'k.', markersize=8, alpha=0.3, label=r'$[XX]$')
ax2.plot(w0, sol[:, -1, 3], 'kx', markersize=5, alpha=0.5, label=r'$[XY]$')
ax2.plot(w0, sol[:, -1, 4], 'ks', markersize=5, alpha=0.5, label=r'$[YY]$')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.minorticks_on()
legend1 = ax1.legend(frameon=False, loc=1)
legend2 = ax2.legend(frameon=False, loc=1)
#ax1.set_yticks(np.arange(0, 1.01, step=0.1), fontsize=13)
#ax1.set_xticks(np.arange(0, 1.01, step=0.1), fontsize=13)
# plt.savefig("FILENAME.pdf", bbox_inches="tight")
plt.show()


if False:
    if denN == 9:
        # Check connectivity of the network, so there are no loose nodes
        if fsum(moments[0:3]) == 1 and fsum(moments[3:]) == 1:
            # Integrate the system for all different noise rates in w0
            for noise in range(len(w0)):
                sol[noise, :, :] = odeint(Dynamics3, moments, t, args=(w0[noise], w2, a, d))
                print(str(noise) + ' / ' + str(len(w0)))

            # Saving our data
            if doSaveSol:
                SaveArray(sol, filename)

            # Plot only the last value corresponding to each noise rate
            ax.plot(w0, sol[:, -1, 0:3], '.', markersize=3, alpha=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.minorticks_on()
            plt.yticks(np.arange(0, 1.01, step=0.1), fontsize=13)
            plt.xticks(np.arange(0, 1.01, step=0.1), fontsize=13)
            # plt.savefig("FILENAME.pdf", bbox_inches="tight")
            plt.show()

        else:
            print('ERROR: densities do no add up / network not connected')

    else:
        print('ERROR: M not in {5,9}')
