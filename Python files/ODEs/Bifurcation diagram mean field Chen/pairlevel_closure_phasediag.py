'''
A phase diagram is computed for the 2 or 3 state discrete adaptive network with pair level closure approximation
# This script is very similar to pairlevel_closure_bifurcation.py, but
# It contains an extra for loop for the a-rates and plots the results differently.

'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import isclose, fsum
# Import our dynamics ODE's
from pairlevel_closure import Dynamics2, Dynamics3


# Definition of the initial conditions
# If M = 2: moments = [X Y XX XY YY]
# If M = 3: moments = [X, Y, Z, XX, XY, YY, YZ, ZZ, ZX]

moments = [0.6, 0.4, 0.3, 0.5, 0.2]
#moments = [0.4, 0.35, 0.25, 0.1, 0.2, .15, .05, .3, .2]
denN = len(moments)
# Check if M = 2 or M = 3
if denN == 5 or denN == 9:
    # Definition of the initial conditions
    Nw0 = 11
    Na = 8
    w0Max = 5
    aMax = 7
    tMax = 1000
    tSteps = 2
    w2 = 1        # Chen uses 0.2
    d = 1         # Chen uses 0.1
    eps = 1e-5      # relative tolerance for determining if the state is ordered or disordered
    fig, ax = plt.subplots(1, 1)
    # Define the range of w0 rates for which to compute the bifurcation
    w0 = np.linspace(0, w0Max, Nw0)
    a = np.linspace(0, aMax, Na)
    t = np.linspace(0, tMax, tSteps)
#    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    if denN == 5:
        # Check connectivity of the network, so there are no loose nodes
        if fsum(moments[0:2]) == 1 and fsum(moments[2:]) == 1:
            # Integrate the system for all different noise rates in w0
            for noise in range(len(w0)):
                for rate in range(len(a)):
                    sol = odeint(Dynamics2, moments, t, args=(w0[noise], w2, a[rate], d))
                    if isclose(sol[-1, 0], sol[-1, 1], rel_tol=eps):
                        # Plot a square if disordered, otherwise a circle
                        ax.plot(w0[noise], a[rate], 'ks', markersize=5, alpha=1)
                    else:
                        ax.plot(w0[noise], a[rate], 'kx', markersize=5, alpha=1)
                    print('DONE: w0 =' + str(w0[noise]) + '  a = ' + str(a[rate]))
#            ax.set_xlim(0, w0Max)
#            ax.set_ylim(0, aMax)
#            ax.minorticks_on()
#            plt.yticks(np.arange(0, 1.01, step=0.1), fontsize=13)
#            plt.xticks(np.arange(0, 1.01, step=0.1), fontsize=13)
#            # plt.savefig("FILENAME.pdf", bbox_inches="tight")
#            plt.show()

        else:
            print('ERROR: densities do no add up / network not connected')

    if denN == 9:
        # Check connectivity of the network, so there are no loose nodes
        if fsum(moments[0:3]) == 1 and fsum(moments[3:]) == 1:
            # Integrate the system for all different noise rates in w0
            for noise in range(len(w0)):
                for rate in range(len(a)):
                    sol = odeint(Dynamics3, moments, t, args=(w0[noise], w2, a[rate], d))

        else:
            print('ERROR: densities do no add up / network not connected')

else:
    print('ERROR: M not in {5,9}')


def l_func(s):
    return np.sqrt(8*s)


#plt.yticks(np.arange(0, 1.01, step=0.1), fontsize=13)
#plt.xticks(np.arange(0, 1.01, step=0.1), fontsize=13)


s = np.linspace(0, 5, 10000)
l = l_func(s)

ax.minorticks_on()
ax.plot(s, l, 'k-', linewidth=3)
ax.set_xlim(0, 5)
ax.set_ylim(0, 7)
ax.set_xlabel(r"$s=\frac{\eta}{\sigma_d}$", fontsize=12)
ax.set_ylabel(r"$\ell=\frac{\alpha}{\beta}$", fontsize=12, rotation=0, labelpad=20)
ax.fill_between(s, l, color='k', alpha=0.1)
ax.annotate(r'ordered', xy=(0.94, 5.4), xytext=(0.94, 5.4), fontsize=12)
ax.annotate(r'disordered', xy=(2.775, 2.4), xytext=(2.775, 2.4), fontsize=12)
# for legend
ax.plot(-1, -1, 'ks', markersize=5, alpha=1, label='disordered')
ax.plot(-1, -1, 'kx', markersize=5, alpha=1, label='ordered')
legend = plt.legend(title='final solution', frameon=True, loc=4)
plt.setp(legend.get_title(), fontsize=12)


# fig.savefig('phase_diag_M2_sims.pdf')

plt.show()
