'''
Plotting bifurcation diagrams of the two state adaptive network model (fig 3.3)
'''


import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter
#import seaborn as sns
# sns.set()
# sns.set_style("white")
#sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3.5})
#plt.rcParams.update({"xtick.bottom": True, "ytick.left": True})


def ordered(s, l):
    Xpart = 0.5*np.sqrt(1-8*s/l**2)
    if Xpart.imag != 0:
        return [np.nan, np.nan]
    else:
        return [0.5 + Xpart, 0.5-Xpart]


def disordered_firstM(s, l):
    XY = l/4
    if (l**2+8*s) == 0:
        return [np.nan, np.nan, np.nan]
    else:
        XX = l/8 * (l**2+4*l+8*s)/(l**2+8*s)
        YY = XX
        return [XX, YY, XY]


def ordered_firstM(s, l):
    if l == 0 or s == 0 or 1-8*s/l**2 < 0:
        return [np.nan, np.nan, np.nan]
    else:
        XY = 2*s/l
        XX = s/l * (1+2/l) * (1+np.sqrt(1-8*s/l**2))/(1-np.sqrt(1-8*s/l**2))
        YY = s/l * (1+2/l) * (1-np.sqrt(1-8*s/l**2))/(1+np.sqrt(1-8*s/l**2))
        return [XX, YY, XY]


def l_func(s):
    return np.sqrt(8*s)


plot_ZM = False
plot_FM = True
plot_s_vs_l = False


if plot_ZM or plot_FM:
    l = [0.5]
    s = np.linspace(0.001, 2, 100)
    Xval = np.zeros((len(s), len(l), 2))
    FM_o = np.zeros((len(s), len(l), 3))
    FM_d = np.zeros((len(s), len(l), 3))
    for Lratio in range(len(l)):
        for Sratio in range(len(s)):
            Xval[Sratio, Lratio, 0] = ordered(s[Sratio], l[Lratio])[0]
            Xval[Sratio, Lratio, 1] = ordered(s[Sratio], l[Lratio])[1]
            FM_o[Sratio, Lratio, 0] = ordered_firstM(s[Sratio], l[Lratio])[0]
            FM_o[Sratio, Lratio, 1] = ordered_firstM(s[Sratio], l[Lratio])[1]
            FM_o[Sratio, Lratio, 2] = ordered_firstM(s[Sratio], l[Lratio])[2]
            FM_d[Sratio, Lratio, 0] = disordered_firstM(s[Sratio], l[Lratio])[0]
            FM_d[Sratio, Lratio, 1] = disordered_firstM(s[Sratio], l[Lratio])[1]
            FM_d[Sratio, Lratio, 2] = disordered_firstM(s[Sratio], l[Lratio])[2]

    s2 = [0.5]
    l2 = np.linspace(0.001, 10, 100)
    Xval2 = np.zeros((len(l2), len(s2), 2))
    FM_o2 = np.zeros((len(l2), len(s2), 3))
    FM_d2 = np.zeros((len(l2), len(s2), 3))
    for Srat in range(len(s2)):
        for Lrat in range(len(l2)):
            Xval2[Lrat, Srat, 0] = ordered(s2[Srat], l2[Lrat])[0]
            Xval2[Lrat, Srat, 1] = ordered(s2[Srat], l2[Lrat])[1]
            FM_o2[Lrat, Srat, 0] = ordered_firstM(s2[Srat], l2[Lrat])[0]
            FM_o2[Lrat, Srat, 1] = ordered_firstM(s2[Srat], l2[Lrat])[1]
            FM_o2[Lrat, Srat, 2] = ordered_firstM(s2[Srat], l2[Lrat])[2]
            FM_d2[Lrat, Srat, 0] = disordered_firstM(s2[Srat], l2[Lrat])[0]
            FM_d2[Lrat, Srat, 1] = disordered_firstM(s2[Srat], l2[Lrat])[1]
            FM_d2[Lrat, Srat, 2] = disordered_firstM(s2[Srat], l2[Lrat])[2]

    if plot_FM:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5))
        ax1.minorticks_on()
        ax1.tick_params(labelsize=12)
        branch1 = np.where(s < 0.5)
        branch2 = np.where(s >= 0.5)
        ax1.plot(s[branch1], FM_o[branch1, 0, 0][0, :], 'k-',
                 linewidth=3, label=(r'$[XX]$ / $[YY]$, s'))
        ax1.plot(s[branch1], FM_o[branch1, 0, 1][0, :], 'k-',
                 linewidth=3)  # , label=(r'$[YY]$, s'))

    #    ax1.plot(s, FM_o[:, 1, 0], 'g-.', linewidth = 2, label=(r'$\ell=1$'))
    #    ax1.plot(s, FM_o[:, 1, 1], 'g-.', linewidth = 2)
    #    ax1.plot(s, FM_o[:, 1, 2], 'g-.', linewidth = 2)
    #    ax1.plot(s, FM_o[:, 2, 0], 'b--', linewidth = 2, label=(r'$\ell=2$'))
    #    ax1.plot(s, FM_o[:, 2, 1], 'k--', linewidth = 2)
    #    ax1.plot(s, FM_o[:, 2, 2], 'r--', linewidth = 2)
        ax1.plot(s[branch1], FM_d[branch1, 0, 0][0, :], 'k:',
                 linewidth=3, label=(r'$[XX]$ / $[YY]$, u'))
        ax1.plot(s[branch1], FM_d[branch1, 0, 1][0, :], 'k:',
                 linewidth=3)  # , label=(r'$[YY]$, u'))
        ax1.plot(s[branch1], FM_o[branch1, 0, 2][0, :], 'r-', linewidth=3, label=(r'$[XY]$, s'))

        ax1.plot(s[branch1], FM_d[branch1, 0, 2][0, :], 'r:', linewidth=3, label=(r'$[XY]$, u'))
        ax1.plot(s[branch2], FM_d[branch2, 0, 0][0, :], 'k-', linewidth=3)
        ax1.plot(s[branch2], FM_d[branch2, 0, 1][0, :], 'k-', linewidth=3)
        ax1.plot(s[branch2], FM_d[branch2, 0, 2][0, :], 'r-', linewidth=3)

        ax1.axvline(x=l[0]**2/8, color='k', linestyle='--', alpha=0.2, linewidth=3)
        ax1.annotate(r'$s=\frac{\ell^2}{8}$', xy=(0.55, 1.68), xytext=(0.55, 1.68), fontsize=14)

    #    ax1.plot(s, FM_d[:, 1, 0], 'g-.', linewidth = 2, label=(r'$\ell=1$'))
    #    ax1.plot(s, FM_d[:, 1, 1], 'g-.', linewidth = 2)
    #    ax1.plot(s, FM_d[:, 1, 2], 'g-.', linewidth = 2)
    #    ax1.plot(s, FM_d[:, 2, 0], 'b--', linewidth = 2, label=(r'$\ell=2$'))
    #    ax1.plot(s, FM_d[:, 2, 1], 'k--', linewidth = 2)
    #    ax1.plot(s, FM_d[:, 2, 2], 'r--', linewidth = 2)

        ax1.set_xlim(0, 0.2)
        ax1.set_ylim(0, 1.25)
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax1.legend(loc='upper right', frameon=False, fontsize=12)
        ax1.set_xlabel(r"$s=\frac{\eta}{\sigma_d}$", fontsize=14)
        ax1.set_ylabel(r"density", fontsize=14)

        ax2.tick_params(labelsize=12)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_tick_params(labelright=True, labelleft=False)
        ax2.yaxis.tick_right()
        ax2.minorticks_on()
        branch1 = np.where(l2 < 4)
        branch2 = np.where(l2 >= 4)

        ax2.plot(l2, FM_o2[:, 0, 0], 'k-', linewidth=3, label=(r'$[XX]$ / $[YY]$, s'))
        ax2.plot(l2, FM_o2[:, 0, 1], 'k-', linewidth=3)
    #    ax2.plot(l2, FM_o2[:, 1, 0], 'g-.', linewidth = 2, label=(r'$\ell=1$'))
    #    ax2.plot(l2, FM_o2[:, 1, 1], 'g-.', linewidth = 2)
    #    ax2.plot(l2, FM_o2[:, 1, 2], 'g-.', linewidth = 2)
    #    ax2.plot(l2, FM_o2[:, 2, 0], 'b--', linewidth = 2, label=(r'$\ell=2$'))
    #    ax2.plot(l2, FM_o2[:, 2, 1], 'k--', linewidth = 2)
    #    ax2.plot(l2, FM_o2[:, 2, 2], 'r--', linewidth = 2)

        ax2.plot(l2[branch1], FM_d2[branch1, 0, 0][0, :], 'k-', linewidth=3)
        ax2.plot(l2[branch1], FM_d2[branch1, 0, 1][0, :], 'k-', linewidth=3)
        ax2.plot(l2[branch2], FM_d2[branch2, 0, 0][0, :], 'k:',
                 linewidth=3, label=(r'$[XX]$ / $[YY]$, u'))
        ax2.plot(l2[branch2], FM_d2[branch2, 0, 1][0, :], 'k:', linewidth=3)
        ax2.plot(l2[branch1], FM_d2[branch1, 0, 2][0, :], 'r-', linewidth=3, label=(r'$[XY]$, s'))
        ax2.plot(l2[branch2], FM_d2[branch2, 0, 2][0, :], 'r:', linewidth=3, label=(r'$[XY]$, u'))
        ax2.plot(l2[branch1], FM_o2[branch1, 0, 2][0, :], 'r-', linewidth=3)
        ax2.plot(l2[branch2], FM_o2[branch2, 0, 2][0, :], 'r-', linewidth=3)

        ax2.axvline(x=np.sqrt(8*s2[0]), color='k', linestyle='--', alpha=0.2, linewidth=3)
        ax2.annotate(r'$\ell=2\sqrt{2s}$', xy=(4.15, 3.55), xytext=(4.15, 3.55), fontsize=14)

    #    ax2.plot(l2, FM_d2[:, 1, 0], 'g-.', linewidth = 2, label=(r'$\ell=1$'))
    #    ax2.plot(l2, FM_d2[:, 1, 1], 'g-.', linewidth = 2)
    #    ax2.plot(l2, FM_d2[:, 1, 2], 'g-.', linewidth = 2)
    #    ax2.plot(l2, FM_d2[:, 2, 0], 'b--', linewidth = 2, label=(r'$\ell=2$'))
    #    ax2.plot(l2, FM_d2[:, 2, 1], 'k--', linewidth = 2)
    #    ax2.plot(l2, FM_d2[:, 2, 2], 'r--', linewidth = 2)

        ax2.set_ylim(0, 3.5)
        ax2.set_xlim(0, 5)
        ax2.legend(loc='upper left', frameon=False, fontsize=12)
        ax2.set_xlabel(r"$\ell=\frac{\alpha}{\beta}$", fontsize=14)
        ax2.set_ylabel(r"density", fontsize=14)
    #    fig.savefig('bifurcation_first_s_2_l_2.pdf')
        plt.show()

    if plot_ZM:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5))
        ax1.minorticks_on()
        ax1.tick_params(labelsize=12)
        ax1.plot(s, Xval[:, 0, 0], 'k--', linewidth=1, label=(r'$\ell=\frac{1}{2}$'))
        ax1.plot(s, Xval[:, 0, 1], 'k--', linewidth=1)
        ax1.plot(s, Xval[:, 1, 0], 'k-.', linewidth=2, label=(r'$\ell=1$'))
        ax1.plot(s, Xval[:, 1, 1], 'k-.', linewidth=2)
        ax1.plot(s, Xval[:, 2, 0], 'k-', linewidth=3, label=(r'$\ell=2$'))
        ax1.plot(s, Xval[:, 2, 1], 'k-', linewidth=3)
        ax1.plot(np.linspace(0.5, 2, 100), np.ones((100))/2, 'k-', linewidth=3, alpha=1)
        ax1.plot(np.linspace(0.12, 0.5, 100), np.ones((100))/2, 'k-.', linewidth=2, alpha=1)
        ax1.plot(np.linspace(0.03, 0.12, 100), np.ones((100))/2, 'k--', linewidth=1, alpha=1)
        ax1.plot(np.linspace(0, 0.03, 100), np.ones((100))/2, 'k:', linewidth=1, alpha=1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 0.65)
        ax1.legend(loc='upper right', frameon=False, fontsize=12)
        ax1.set_xlabel(r"$s=\frac{\eta}{\sigma_d}$", fontsize=14)
        ax1.set_ylabel(r"$[X]$  /  $[Y]$", fontsize=14)

        ax2.tick_params(labelsize=12)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_tick_params(labelright=True, labelleft=False)
        ax2.yaxis.tick_right()
        ax2.minorticks_on()
        ax2.plot(l2, Xval2[:, 0, 0], 'k-', linewidth=3, label=(r'$s=\frac{1}{2}$'))
        ax2.plot(l2, Xval2[:, 0, 1], 'k-', linewidth=3)
        ax2.plot(l2, Xval2[:, 1, 0], 'k-.', linewidth=2, label=(r'$s=1$'))
        ax2.plot(l2, Xval2[:, 1, 1], 'k-.', linewidth=2)
        ax2.plot(l2, Xval2[:, 2, 0], 'k--', linewidth=1, label=(r'$s=2$'))
        ax2.plot(l2, Xval2[:, 2, 1], 'k--', linewidth=1)
        ax2.plot(np.linspace(4, 12, 100), np.ones((100))/2, 'k:', linewidth=1, alpha=1)
        ax2.plot(np.linspace(2.83, 4, 100), np.ones((100))/2, 'k--', linewidth=1, alpha=1)
        ax2.plot(np.linspace(2, 2.83, 100), np.ones((100))/2, 'k-.', linewidth=2, alpha=1)
        ax2.plot(np.linspace(0, 2, 100), np.ones((100))/2, 'k-', linewidth=3, alpha=1)

        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 10)
        ax2.legend(loc='upper left', frameon=False, fontsize=12)
        ax2.set_xlabel(r"$\ell=\frac{\alpha}{\beta}$", fontsize=14)
        ax2.set_ylabel(r"$[X]$  /  $[Y]$", fontsize=14)
        # fig.savefig('bifurcation_zeroth.pdf')
        plt.show()


if plot_s_vs_l:
    s = np.linspace(0, 5, 100)
    l = l_func(s)
    fig, ax = plt.subplots(1, 1)
    ax.minorticks_on()
    ax.plot(s, l, 'k-', linewidth=3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 7)
    ax.set_xlabel(r"$s=\frac{\eta}{\sigma_d}$", fontsize=12)
    ax.set_ylabel(r"$\ell=\frac{\alpha}{\beta}$", fontsize=12, rotation=0, labelpad=20)
    ax.fill_between(s, l, color='k', alpha=0.1)
    ax.annotate(r'ordered', xy=(1, 5), xytext=(1, 5), fontsize=12)
    ax.annotate(r'disordered', xy=(3, 2), xytext=(3, 2), fontsize=12)
    # fig.savefig('phase_diag_M2.pdf')

    plt.show()
