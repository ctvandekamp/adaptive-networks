'''
Density vs time computations/plots of the discrete 2 or 3 state adaptive network model with pair level closure approximation
'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import fsum
import json
# Import our dynamics ODE's
from pairlevel_closure import Dynamics2, Dynamics3


# nNodes = 300
# nLinks = 300
# nX = 35
# nY = nNodes - nX
# nXX = 100
# nYY = 100
# nXY = nLinks - nXX - nYY

# moments = [nX/nNodes, nY/nNodes, nXX/nLinks, nXY/nLinks, nYY/nLinks]

# Definition of the initial conditions
# If M = 2: moments = [X Y XX XY YY]
# If M = 3: moments = [X, Y, Z, XX, XY, YY, YZ, ZZ, ZX]

moments1 = [0.7, 0.3, 1.2, 1.0, 0.7]
moments2 = [0.6, 0.4, 0.3, 0.4, 0.3]
# moments = [0.4, 0.35, 0.25, 0.1, 0.2, .15, .05, .3, .2]
denN = len(moments1)
# Check if M = 2 or M = 3
if denN == 5 or denN == 9:
    tMax = 25
    tSteps = 500
    t = np.linspace(0, tMax, tSteps)
    w0 = 0.65
    w2 = 0.2        # Chen uses 0.2
    a = 0.5         # Chen uses 0.5
    d = 0.1         # Chen uses 0.1
    doPhasePortrait = 0  # only in M=2 case

    if denN == 5:
        # Check connectivity of the network, so there are no loose nodes which cannot interact
        if fsum(moments1[0:2]) == 1:
            # Integration of the system
            sol = odeint(Dynamics2, moments1, t, args=(w0, w2, a, d,))
            sol2 = odeint(Dynamics2, moments2, t, args=(w0, w2, a, d,))

# Plotting a single solution versus time
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.4, 9.1))
ax1.plot(t, sol[:, 0], 'k-', label=r'$[X]$')
ax1.plot(t, sol[:, 1], 'k--', label=r'$[Y]$')
# Plotting of the first order moments
ax2.plot(t, sol[:, 2], 'k-', label=r'$[XX]$')
ax2.plot(t, sol[:, 3], 'k--', label=r'$[XY]$')
ax2.plot(t, sol[:, 4], 'k:', label=r'$[YY]$')
#                   np.transpose(np.asarray([sol[:, 5], sol[:, 5], sol[:, 5]]))))
ax1.set_ylim(0, 1)
ax1.set_xlim(0, tMax)
legend1 = ax1.legend(frameon=False, loc=2)

ax2.set_xlim(0, tMax)
ax2.set_ylim(0, 3)
legend2 = ax2.legend(frameon=False, loc=2)

ax1.set_xlabel(r'$\tau$', fontsize=12)
ax1.set_ylabel('density', fontsize=12)
ax1.minorticks_on()

ax2.yaxis.set_label_position("right")
ax2.yaxis.set_tick_params(labelright=True, labelleft=False)
ax2.yaxis.tick_right()

ax2.set_xlabel(r'$\tau$', fontsize=12)
ax2.set_ylabel('density', fontsize=12)
ax2.minorticks_on()


ax3.plot(t, sol2[:, 0], 'k-', label=r'$[X]$')
ax3.plot(t, sol2[:, 1], 'k--', label=r'$[Y]$')
# Plotting of the first order moments
ax4.plot(t, sol2[:, 2], 'k-', label=r'$[XX]$')
ax4.plot(t, sol2[:, 3], 'k--', label=r'$[XY]$')
ax4.plot(t, sol2[:, 4], 'k:', label=r'$[YY]$')
#                   np.transpose(np.asarray([sol[:, 5], sol[:, 5], sol[:, 5]]))))
ax3.set_ylim(0, 1)
ax3.set_xlim(0, tMax)
legend3 = ax3.legend(frameon=False, loc=2)

ax4.set_xlim(0, tMax)
ax4.set_ylim(0, 3)
legend4 = ax4.legend(frameon=False, loc=2)

ax3.set_xlabel(r'$\tau$', fontsize=12)
ax3.set_ylabel('density', fontsize=12)
ax3.minorticks_on()

ax4.yaxis.set_label_position("right")
ax4.yaxis.set_tick_params(labelright=True, labelleft=False)
ax4.yaxis.tick_right()

ax4.set_xlabel(r'$\tau$', fontsize=12)
ax4.set_ylabel('density', fontsize=12)
ax4.minorticks_on()

#            ax3.plot(sol[:, 0], sol[:, 1])
#
#            ax3 = fig.add_subplot(133, projection='3d')
#            ax3.plot(sol[:, 2], sol[:, 3], sol[:, 4])
#            ax3.set_title("Phase Portrait")
#            ax3.set_xlabel("X")
#            ax3.set_ylabel("Y")
#            ax3.set_ylim(0, 1)
#            ax3.set_xlim(0, 1)
#            ax3.set_zlim(0, 1)

# plt.savefig("discrete_double_run_2.pdf")
plt.show()


#        else:
#            print('ERROR: densities do not add up / network not connected')
if False:
    if denN == 9:
        # Check connectivity of the network, so there are no loose nodes which cannot interact
        if fsum(moments[0:3]) == 1 and fsum(moments[3:9]) == 1:
            # Integration of the system
            sol = odeint(Dynamics3, moments, t, args=(w0, w2, a, d,))
            # The following line renormalizes the link densities. They do not add up to 1 anymore,
            # since the parameters a and d allow link creation in time.
            sol[:, 3:] = sol[:, 3:]/np.transpose(np.tile(np.sum(sol[:, 3:], axis=1), (6, 1)))

            # Plotting a single solution versus time
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7.5))
            ax1.plot(t, sol[:, 0:3])
            # Plotting of the first order moments
            ax2.plot(t, np.asarray(sol[:, 3:9]))
            ax1.set_ylim(0, 1)
            ax1.set_xlim(0, tMax)
            ax1.legend(('X', 'Y', 'Z'))
            ax2.set_xlim(0, tMax)
            ax2.set_ylim(0, 1)
            ax2.legend(('XX', 'XY', 'YY', 'YZ', 'ZZ', 'ZY'))
            ax1.minorticks_on()
            # plt.savefig("FILENAME.pdf", bbox_inches="tight")
            plt.show()

        else:
            print('ERROR: densities do not add up / network not connected')

    # dumpData = {"moments": moments, "tMax": tMax, "tSteps": tSteps,
    #             "w0": w0, "w2": w2, "a": a, "d": d, "sol": sol}
    # print(dumpData)
    #
    # with open("singlerun denN={denN} tMax={tMax} w0={w0} a={a}", "w+") as write_file:
    #     json.dump(dumpData, write_file)
