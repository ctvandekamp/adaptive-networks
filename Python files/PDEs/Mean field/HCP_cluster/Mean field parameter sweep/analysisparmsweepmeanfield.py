'''
This script is able to analyse a certain parameter sweep in a mean field adaptive network model
and creates the figures used in the thesis.
The part under ''ANALYSIS'' imports the data.
One can choose which part to run by replacing if False by if True.
'''


import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  # to specify number of digits in ticks
from astropy import modeling
from basic_units import radians
from matplotlib import rcParams
from sklearn.metrics import r2_score
rcParams.update({'figure.autolayout': True})
from scipy.optimize import curve_fit
import scipy.stats as stats  # for initialising als normal distribution


"""
Functions for analysis of the parameter sweep
"""


def read_data(filepath):
    """
    Importing all .dat files in a single dictionary.
    The keys contain the sytem parameters in
    eta_sigma_alpha_beta format. Each key value is a
    dictionary with all system parameters, initial
    condition and states and links.
    """
    filepath = str(filepath)
    data = {}
    paramfile = open(filepath, 'r')
    for line in paramfile:
        line1 = line.replace('\n', '')
        line2 = line1.replace(" ", "_")
        try:
            f = open("data{}_{}.dat".format(str(0), line2), "rb")
            data[line2] = pickle.load(f)
        except FileNotFoundError:
            print("data{}_{}.dat".format(str(0), line2), ' NOT FOUND')
    paramfile.close()
    return data


def meanfunc(states, omega):
    """
    This function computes the mean of a distribution
    """
    probabilities = states/states.sum()
    mu = omega.dot(probabilities)
    return mu


def variance(states, omega):
    """
    This function computes the variance of a distribution.
    NOTE: IC is given with STD
    """
#    probabilities = states/states.sum()
#    secondmoment = np.power(omega, 2).dot(probabilities)
#    var = secondmoment - mean(states, omega)**2

    var = np.sum(states*(omega-np.average(omega, weights=states))**2) / np.sum(states)

    return var


"""
ANALYSIS
"""

if True:
    data = read_data('parameters.txt')

    omega = np.linspace(-np.pi, np.pi, num=data['0.0_1.0_1.0']
                        ['steps']['x_steps'], endpoint=False)        # endpoint was false
    for key in data.keys():
        data[key]['final_variance'] = variance(data[key]['states_t%100'][-1], omega)


#####################################################################
        # FOR PRESENTATION
######################################################################

"""
Fitting a Gaussian curve with custom amplitude, mean and std, generates plot which is currently in the thesis
Fitting a Lorentzian with custom amplitude, position of the peak x_0 and full width at half maximum fwhm
"""
if False:
    omega_axis = [val*radians for val in omega]
    dataset1 = data['0.01_1.0_1.0']['states_t%100'][-1]
    dataset2 = data['0.03_1.0_1.0']['states_t%100'][-1]
    dataset3 = data['0.06_1.0_1.0']['states_t%100'][-1]
    fitter = modeling.fitting.LevMarLSQFitter()
    lorentz = modeling.models.Lorentz1D()
    fitted_model1_lorentz = fitter(lorentz, omega, dataset1)
    fitted_model2_lorentz = fitter(lorentz, omega, dataset2)
    fitted_model3_lorentz = fitter(lorentz, omega, dataset3)

    fig = plt.figure(figsize=(8, 5))  # figsize=(1.75*6.4,4.8))

    ax2 = fig.add_subplot(1, 1, 1)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax2.set_xlabel(r'$x$', fontsize=16)
    ax2.set_ylabel(r'$f\, (x;t)$', fontsize=16, rotation=0, labelpad=30)
    ax2.set_xlim((-np.pi, np.pi))
    ax2.set_ylim((0, 2.2))
    ax2.plot(omega_axis, dataset1, color='k', linewidth=1,
             linestyle='-', alpha=0.5, label=r'$0.01$')
    ax2.plot(omega_axis, fitted_model1_lorentz(omega), color='k',
             linewidth=1.5, linestyle='--', alpha=1, label=r'$0.01$ LSQ C')
    ax2.plot(omega_axis, dataset2, color='k', linewidth=1,
             linestyle='-', alpha=0.5, label=r'$0.03$')
    ax2.plot(omega_axis, fitted_model2_lorentz(omega), color='k',
             linewidth=2, linestyle='--', alpha=1, label=r'$0.03$ LSQ C')
    ax2.plot(omega_axis, dataset3, color='k', linewidth=2,
             linestyle='-', alpha=0.5, label=r'$0.06$')
    ax2.plot(omega_axis, fitted_model3_lorentz(omega), color='k',
             linewidth=3, linestyle='--', alpha=1, label=r'$0.06$ LSQ C')
    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(1/4))

    legend2 = ax2.legend(title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False)
    plt.setp(legend2.get_title(), fontsize=18)
    plt.show()
    fig.savefig('cont_mean_field_fits_6.pdf')


"""
Computing variance and plotting
"""


def analyticroot(n, a, b):
    return np.pi**2/3 - a*np.sqrt(b - n)


if False:
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(1, 1, 1)
#    sub_axes = plt.axes([.65, .7, .27, .22])
#    ax1.title.set_text('Variance')
    ax1.set_xlabel(r'$\frac{\eta}{\sigma \langle k \rangle^2}$', fontsize=17)
    # labelpad increases distance label to axis
    ax1.set_ylabel(r'variance', fontsize=14, rotation=0, labelpad=40)
    ax1.set_ylim((0, 6))
    ax1.set_xlim((0, 0.15))
#    ax1.axvline(x=0.068, color='k', linestyle = '--', alpha = 0.2)
    ax1.axvline(x=0.072, color='k', linestyle='--', alpha=0.2)
#    ax1.annotate(r'$0.068$', xy = (0.054, 0.3), xytext=(0.054, 0.3), fontsize=14)
    ax1.annotate(r'$0.072$', xy=(0.073, 0.3), xytext=(0.073, 0.3), fontsize=14)
    ax1.axhline(y=np.pi**2/3, color='k', linestyle='--', alpha=0.2)
    ax1.annotate(r'$\frac{\pi^2}{3}$', xy=(0.003, np.pi**2/3+0.22),
                 xytext=(0.003, np.pi**2/3+0.22), fontsize=17)
    # specify number of digits in ticks
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.tick_params(axis='both', which='major', labelsize=14)


#    sub_axes.axvline(x=0.068, color='k', linestyle = '--', alpha = 0.2)
#    sub_axes.axvline(x=0.072, color='k', linestyle = '--', alpha = 0.2)
#    sub_axes.axhline(y=np.pi**2/3, color='k', linestyle = '--', alpha = 0.2)

    xdata = np.zeros(len(data.keys()))
    ydata = np.zeros(len(data.keys()))
    i = 0
    for run in data.keys():
        states = data[run]['states_t%100']
        n = data[run]['system']['n']
        varb = variance(states[0, :], omega)
        vare = variance(states[-1, :], omega)
       # ax1.plot(n, varb, 'b.')
        ax1.plot(n, vare, 'k.', alpha=0.2, markersize=8)
#        sub_axes.plot(n, vare, 'k.',alpha=0.2, markersize=8)
        xdata[i] = n
        ydata[i] = vare
        i += 1
    xdata2 = xdata[72:]
    xdata = xdata[:69]
    ydata = ydata[:69]
    # fitting a sqrt
    popt, pcov = curve_fit(analyticroot, xdata, ydata, bounds=(0.068, [100, 0.075]))
#    ax1.plot(xdata, analyticroot(xdata, *popt), 'k--', linewidth=3)
    ax1.plot(xdata2, np.ones(len(xdata2))*np.pi**2/3, 'k--', linewidth=3)
#    sub_axes.plot(xdata, analyticroot(xdata, *popt), 'k--')
#    sub_axes.plot(xdata2, np.ones(len(xdata2))*np.pi**2/3, 'k--')
#    sub_axes.set_xlim((0.063,0.077))
#    sub_axes.set_ylim((2.5,3.5))
    ax1.minorticks_on()
#    plt.show()
    fig.savefig('cont_mean_field_variance_2.pdf')
    r2_sqrt = r2_score(ydata, analyticroot(xdata, *popt))

    plt.show()

"""
Plotting final distributions
"""

if False:
    palette = plt.get_cmap('tab10')
    keys = [*data]
    fig = plt.figure()
    omega_axis = [val*radians for val in omega]
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$f\, (x;t)$', fontsize=16, rotation=0, labelpad=28)
    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((0, 1.5))
    ax.tick_params(axis='both', which='major', labelsize=14)

    greys = True
    if greys:
        #        ax.plot(omega_axis, data['0.015_1.0_1.0']['states_t%100'][-1], color = 'k', linewidth = 3, linestyle = ':', alpha = 1, label = r'$0.015$', xunits=radians)
        #        ax.plot(omega_axis, data['0.025_1.0_1.0']['states_t%100'][-1], color = 'k', linewidth = 3,  linestyle = '--', alpha = 1, label = r'$0.025$', xunits=radians)
        #        ax.plot(omega_axis, data['0.06_1.0_1.0']['states_t%100'][-1], color = 'k', linewidth = 1.5, linestyle = '-.', alpha = 1, label = r'$0.060$', xunits=radians)
        ax.plot(omega_axis, data['0.1_1.0_1.0']['states_t%100'][-1], color='k',
                linewidth=3, linestyle='-', alpha=1, label=r'$0.100$', xunits=radians)
#        ax.plot(omega_axis, data['0.0_1.0_1.0']['states_t%100'][0], color = 'k', linewidth = 3.5, linestyle = '-', alpha = 0.35, label = 'initial distribution', xunits=radians)
    else:
        ax.plot(omega_axis, data['0.015_1.0_1.0']['states_t%100'][-1], color=palette(0),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.015$', xunits=radians)
        ax.plot(omega_axis, data['0.025_1.0_1.0']['states_t%100'][-1], color=palette(3),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.025$', xunits=radians)
        ax.plot(omega_axis, data['0.06_1.0_1.0']['states_t%100'][-1], color=palette(2),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.060$', xunits=radians)
        ax.plot(omega_axis, data['0.1_1.0_1.0']['states_t%100'][-1], color=palette(1),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.100$', xunits=radians)
        ax.plot(omega_axis, data['0.0_1.0_1.0']['states_t%100'][0], color='k', linewidth=4,
                linestyle='--', alpha=0.65, label='initial distribution', xunits=radians)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

#    ax.fill_between(omega_axis, 0, data['0.025_1.0_1.0']['states_t%100'][-1], where= abs(omega-np.pi/4) < np.pi/4)
#    ax.annotate(r'$\int_{0}^{\pi/2}\ f(x;t)\ dx$', xy = (np.pi/4, 0.8), xytext=(np.pi/4, 0.8), fontsize=16)
    ax.annotate(r'$\frac{1}{2\pi}$', xy=(-np.pi+0.1, 0.21), xytext=(-np.pi+0.1, 0.21), fontsize=16)

    legend = plt.legend(title=r'$\frac{\eta}{\sigma \langle k \rangle^2}$', frameon=False)
    plt.setp(legend.get_title(), fontsize=16)
    plt.show()
#    fig.savefig('cont_mean_field_stat_sols_1.pdf')


#####################################################################
    # FOR REPORT
######################################################################


"""
Computing variance and plotting
"""


def analyticroot(n, a, b):
    return np.pi**2/3 - a*np.sqrt(b - n)


if False:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    sub_axes = plt.axes([.65, .7, .27, .22])
#    ax1.title.set_text('Variance')
    ax1.set_xlabel(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', fontsize=16)
    # labelpad increases distance label to axis
    ax1.set_ylabel(r'$\sigma_f^2$', fontsize=14, rotation=0, labelpad=10)
    ax1.set_ylim((0, 6))
    ax1.set_xlim((0, 0.15))
    ax1.axvline(x=0.068, color='k', linestyle='--', alpha=0.2)
    ax1.axvline(x=0.072, color='k', linestyle='--', alpha=0.2)
    ax1.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.068$',
                 xy=(0.044, 0.3), xytext=(0.044, 0.3))
    ax1.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.072$',
                 xy=(0.073, 0.3), xytext=(0.073, 0.3))
    ax1.axhline(y=np.pi**2/3, color='k', linestyle='--', alpha=0.2)
    ax1.annotate(r'$\sigma_f^2 = \frac{\pi^2}{3}$', xy=(
        0.003, np.pi**2/3+0.12), xytext=(0.003, np.pi**2/3+0.12))
    # specify number of digits in ticks
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    sub_axes.axvline(x=0.068, color='k', linestyle='--', alpha=0.2)
    sub_axes.axvline(x=0.072, color='k', linestyle='--', alpha=0.2)
    sub_axes.axhline(y=np.pi**2/3, color='k', linestyle='--', alpha=0.2)

    xdata = np.zeros(len(data.keys()))
    ydata = np.zeros(len(data.keys()))
    i = 0
    for run in data.keys():
        states = data[run]['states_t%100']
        n = data[run]['system']['n']
        varb = variance(states[0, :], omega)
        vare = variance(states[-1, :], omega)
       # ax1.plot(n, varb, 'b.')
        ax1.plot(n, vare, 'k.', alpha=0.2, markersize=8)
        sub_axes.plot(n, vare, 'k.', alpha=0.2, markersize=8)
        xdata[i] = n
        ydata[i] = vare
        i += 1
    xdata2 = xdata[72:]
    xdata = xdata[:69]
    ydata = ydata[:69]
    # fitting a sqrt
    popt, pcov = curve_fit(analyticroot, xdata, ydata, bounds=(0.068, [100, 0.075]))
    ax1.plot(xdata, analyticroot(xdata, *popt), 'k--')
    ax1.plot(xdata2, np.ones(len(xdata2))*np.pi**2/3, 'k--')
    sub_axes.plot(xdata, analyticroot(xdata, *popt), 'k--')
    sub_axes.plot(xdata2, np.ones(len(xdata2))*np.pi**2/3, 'k--')
    sub_axes.set_xlim((0.063, 0.077))
    sub_axes.set_ylim((2.5, 3.5))
    ax1.minorticks_on()
#    plt.show()
    # fig.savefig('cont_mean_field_variance_zoom.pdf')
    r2_sqrt = r2_score(ydata, analyticroot(xdata, *popt))

    """ Creating the loglog plot for power check """
    # Placing 0.068 at the origin
    xdata = xdata[:]
    variances = np.flip(ydata) * -1 + np.max(ydata)
    variances = variances[:]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
#    ax1.plot(xdata,xdata**0.5 + 12.52,'r.')
    ax1.loglog(xdata, variances, '-k', linewidth=1.25)
    ax1.loglog(xdata, variances, 'xk', markersize=4)
#    ax1.loglog(xdata,np.flip(analyticroot(xdata, *popt))*-1+np.max(analyticroot(xdata, *popt)), 'ks')
#    ax1.loglog(xdata,np.flip(ydata) * -1 + np.pi**2/3, 'k:')
    ax1.set_ylim((0.05, 10))
    ax1.set_xlim((0.001, 0.1))
    ax1.axvline(x=0.068, color='k', linestyle='--', alpha=0.2)
    ax1.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0$',
                 xy=(0.035, 0.066), xytext=(0.035, 0.066))
    ax1.axhline(y=np.pi**2/3, color='k', linestyle='--', alpha=0.2)
    ax1.annotate(r'$\sigma_f^2 = 0$', xy=(0.0012, np.pi**2/3+0.22),
                 xytext=(0.0012, np.pi**2/3+0.22))
    ax1.plot(xdata, xdata**0.5*np.exp(-0.82+np.pi**2/3), 'k:', linewidth=0.5)
    ax1.set_xlabel(r'$0.068-\frac{\eta}{\sigma_c \langle k \rangle^2}$', fontsize=14)
    ax1.set_ylabel(r'$\frac{\pi^2}{3} - \sigma_f^2$', fontsize=14, rotation=0,
                   labelpad=23)   # labelpad increases distance label to axis
#    fig.savefig('cont_mean_field_variance_loglog.pdf')
    plt.show()


"""
Plotting variance versus time
"""
if False:

    # we have to import another data set which contains many more time points
    palette = plt.get_cmap('tab10')
    filepath = str('var_vs_t/parameters.txt')
    paramfile = open(filepath, 'r')
    data_many_t = {}
    for line in paramfile:
        line1 = line.replace('\n', '')
        line2 = line1.replace(" ", "_")
        try:
            f = open("var_vs_t/data_manytpoints_{}_{}.dat".format(str(0), line2), "rb")
            data_many_t[line2] = pickle.load(f)
        except FileNotFoundError:
            print("var_vs_t/data_manytpoints_{}_{}.dat".format(str(0), line2), ' NOT FOUND')
    paramfile.close()

    keys = (['0.1_1.0_1.0', '0.08_1.0_1.0', '0.07_1.0_1.0', '0.06_1.0_1.0',
             '0.04_1.0_1.0', '0.03_1.0_1.0', '0.01_1.0_1.0'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(y=np.pi**2/3, color='k', linestyle='--', alpha=0.2)
    plt.annotate(r'$\sigma_f^2 = \frac{\pi^2}{3}$', xy=(
        10, np.pi**2/3-0.17), xytext=(10, np.pi**2/3-0.17))
    color = 0
    widths = [1.15, 1.15, 1.15, 1.15, 2, 2, 2, 2, 2]
    alphas = np.ones(7)
    styles = ['-', '--', '-.', ':', '-', '--', '-.']
    for key in keys:
        states = data_many_t[key]['states']
        var = np.zeros((states.shape[0]))
        for t in range((states.shape[0])):
            var[t] = variance(states[t, :], omega)
        lab = data_many_t[key]['system']['n']
        if lab == 0.1:
            lab = '0.10'
        if color < 4:
            ax.plot(range(300), var[0:300], 'k', linestyle=styles[color],
                    linewidth=widths[color], alpha=alphas[color], label=lab)  # color = palette(color)
        else:
            ax.plot(range(235), var[0:235], 'k', linestyle=styles[color],
                    linewidth=widths[color], alpha=alphas[color], label=lab)  # color = palette(color)
        color += 1
#    ax.plot(range(300), var[0:300,0], 'k', linewidth = 0.5, alpha = 1, label = data_many_t[key]['system']['n'])
#    ax.plot(range(300), var[0:300,1], 'k', linewidth = 1.0, alpha = 1, label = data_many_t[key]['system']['n'])
#    ax.plot(range(300), var[0:300,2], 'k', linewidth = 1.5, alpha = 1, label = data_many_t[key]['system']['n'])
#    ax.plot(range(300), var[0:300,3], 'k', linewidth = 2.0, alpha = 1, label = data_many_t[key]['system']['n'])
#    ax.plot(range(235), var[0:235,4], 'k', linewidth = 2.5, alpha = 1, label = data_many_t[key]['system']['n'])
#    ax.plot(range(235), var[0:235,5], 'k', linewidth = 3.0, alpha = 1, label = data_many_t[key]['system']['n'])
#    ax.plot(range(235), var[0:235,6], 'k', linewidth = 3.5, alpha = 1, label = data_many_t[key]['system']['n'])
#
#
    legend = plt.legend(loc=4, title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False)
    plt.setp(legend.get_title(), fontsize=16)
    ax.set_ylim((0, 3.5))
    ax.set_xlim((0, 300))
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel(r'$\sigma_f^2$', fontsize=14, rotation=0, labelpad=15)
    plt.show()
    # fig.savefig('cont_mean_field_var_vs_t_BW.pdf')


"""
Plotting final distributions
"""

if False:
    palette = plt.get_cmap('tab10')
    keys = [*data]
    fig = plt.figure()
    omega_axis = [val*radians for val in omega]
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$f\, (x;t)$', fontsize=12, rotation=0, labelpad=20)
    ax.set_xlim((-np.pi, np.pi))
    ax.set_ylim((0, 1.5))

    greys = True
    if greys:
        ax.plot(omega_axis, data['0.015_1.0_1.0']['states_t%100'][-1], color='k',
                linewidth=1, linestyle=':', alpha=1, label=r'$0.015$', xunits=radians)
        ax.plot(omega_axis, data['0.025_1.0_1.0']['states_t%100'][-1], color='k',
                linewidth=1.25,  linestyle='--', alpha=1, label=r'$0.025$', xunits=radians)
        ax.plot(omega_axis, data['0.06_1.0_1.0']['states_t%100'][-1], color='k',
                linewidth=1.5, linestyle='-.', alpha=1, label=r'$0.060$', xunits=radians)
        ax.plot(omega_axis, data['0.1_1.0_1.0']['states_t%100'][-1], color='k',
                linewidth=1.75, linestyle='-', alpha=1, label=r'$0.100$', xunits=radians)
        ax.plot(omega_axis, data['0.0_1.0_1.0']['states_t%100'][0], color='k', linewidth=3.5,
                linestyle='-', alpha=0.35, label='initial distribution', xunits=radians)
    else:
        ax.plot(omega_axis, data['0.015_1.0_1.0']['states_t%100'][-1], color=palette(0),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.015$', xunits=radians)
        ax.plot(omega_axis, data['0.025_1.0_1.0']['states_t%100'][-1], color=palette(3),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.025$', xunits=radians)
        ax.plot(omega_axis, data['0.06_1.0_1.0']['states_t%100'][-1], color=palette(2),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.060$', xunits=radians)
        ax.plot(omega_axis, data['0.1_1.0_1.0']['states_t%100'][-1], color=palette(1),
                linewidth=1.5, linestyle='-', alpha=1, label=r'$0.100$', xunits=radians)
        ax.plot(omega_axis, data['0.0_1.0_1.0']['states_t%100'][0], color='k', linewidth=4,
                linestyle='--', alpha=0.65, label='initial distribution', xunits=radians)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    legend = plt.legend(title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False)
    plt.setp(legend.get_title(), fontsize=16)
    plt.show()
    # fig.savefig('cont_mean_field_final_dist_BW.pdf')

"""
Fitting a Gaussian curve with custom amplitude, mean and std, generates plot which is currently in the thesis
Fitting a Lorentzian with custom amplitude, position of the peak x_0 and full width at half maximum fwhm
"""
if False:
    omega_axis = [val*radians for val in omega]
    dataset1 = data['0.01_1.0_1.0']['states_t%100'][-1]
    dataset2 = data['0.03_1.0_1.0']['states_t%100'][-1]
    dataset3 = data['0.06_1.0_1.0']['states_t%100'][-1]
    fitter = modeling.fitting.LevMarLSQFitter()
    gauss = modeling.models.Gaussian1D()
    lorentz = modeling.models.Lorentz1D()
    fitted_model1_gauss = fitter(gauss, omega, dataset1)
    fitted_model2_gauss = fitter(gauss, omega, dataset2)
    fitted_model3_gauss = fitter(gauss, omega, dataset3)
    fitted_model1_lorentz = fitter(lorentz, omega, dataset1)
    fitted_model2_lorentz = fitter(lorentz, omega, dataset2)
    fitted_model3_lorentz = fitter(lorentz, omega, dataset3)

    fig = plt.figure(figsize=(1.75*6.4, 4.8))
    palette = plt.get_cmap('tab10')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$f\, (x;t)$', fontsize=12, rotation=0, labelpad=20)
    ax1.set_xlim((-np.pi, np.pi))
    ax1.set_ylim((0, 2.2))
    ax1.plot(omega_axis, dataset1, color='k', linewidth=1,
             linestyle='-', alpha=0.25, label=r'$0.01$')
    ax1.plot(omega_axis, fitted_model1_gauss(omega), color='k',
             linewidth=1, linestyle='--', alpha=1, label=r'$0.01$ LSQ G')
    ax1.plot(omega_axis, dataset2, color='k', linewidth=1.5,
             linestyle='-', alpha=0.25, label=r'$0.03$')
    ax1.plot(omega_axis, fitted_model2_gauss(omega), color='k',
             linewidth=1, linestyle='-.', alpha=1, label=r'$0.03$ LSQ G')
    ax1.plot(omega_axis, dataset3, color='k', linewidth=2,
             linestyle='-', alpha=0.25, label=r'$0.06$')
    ax1.plot(omega_axis, fitted_model3_gauss(omega), color='k',
             linewidth=1.5, linestyle=':', alpha=1, label=r'$0.06$ LSQ G')
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel(r'$x$', fontsize=12)
    ax2.set_ylabel(r'$f\, (x;t)$', fontsize=12, rotation=0, labelpad=20)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_tick_params(labelright=True, labelleft=False)
    ax2.yaxis.tick_right()
    ax2.set_xlim((-np.pi, np.pi))
    ax2.set_ylim((0, 2.2))
    ax2.plot(omega_axis, dataset1, color='k', linewidth=1,
             linestyle='-', alpha=0.25, label=r'$0.01$')
    ax2.plot(omega_axis, fitted_model1_lorentz(omega), color='k',
             linewidth=1, linestyle='--', alpha=1, label=r'$0.01$ LSQ C')
    ax2.plot(omega_axis, dataset2, color='k', linewidth=1.5,
             linestyle='-', alpha=0.25, label=r'$0.03$')
    ax2.plot(omega_axis, fitted_model2_lorentz(omega), color='k',
             linewidth=1, linestyle='-.', alpha=1, label=r'$0.03$ LSQ C')
    ax2.plot(omega_axis, dataset3, color='k', linewidth=2,
             linestyle='-', alpha=0.25, label=r'$0.06$')
    ax2.plot(omega_axis, fitted_model3_lorentz(omega), color='k',
             linewidth=1.5, linestyle=':', alpha=1, label=r'$0.06$ LSQ C')
    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(1/4))

    legend1 = ax1.legend(title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False)
    legend2 = ax2.legend(title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False)

    plt.setp(legend1.get_title(), fontsize=16)
    plt.setp(legend2.get_title(), fontsize=16)
    plt.show()
#    fig.savefig('cont_mean_field_G_C_LSQ_BW.pdf')
    fitted_model1_gauss_r2 = r2_score(dataset1, fitted_model1_gauss(omega))
    fitted_model2_gauss_r2 = r2_score(dataset2, fitted_model2_gauss(omega))
    fitted_model3_gauss_r2 = r2_score(dataset3, fitted_model3_gauss(omega))

    fitted_model1_lorentz_r2 = r2_score(dataset1, fitted_model1_lorentz(omega))
    fitted_model2_lorentz_r2 = r2_score(dataset2, fitted_model2_lorentz(omega))
    fitted_model3_lorentz_r2 = r2_score(dataset3, fitted_model3_lorentz(omega))

    print(fitted_model1_gauss, 'r^2 = ', fitted_model1_gauss_r2)
    print(fitted_model2_gauss, 'r^2 = ', fitted_model2_gauss_r2)
    print(fitted_model3_gauss, 'r^2 = ', fitted_model3_gauss_r2)
    print(fitted_model1_lorentz, 'r^2 = ', fitted_model1_lorentz_r2)
    print(fitted_model2_lorentz, 'r^2 = ', fitted_model2_lorentz_r2)
    print(fitted_model3_lorentz, 'r^2 = ', fitted_model3_lorentz_r2)
"""
Plotting these final distributions on loglin and loglog scales
"""
if False:
    omega_axis = [val*radians for val in omega]
    dataset1 = data['0.01_1.0_1.0']['states_t%100'][-1]
    dataset2 = data['0.03_1.0_1.0']['states_t%100'][-1]
    dataset3 = data['0.06_1.0_1.0']['states_t%100'][-1]

    fig = plt.figure(figsize=(1.75*6.4, 4.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(omega_axis, dataset1, color='k', linewidth=1.5,
             linestyle='-', alpha=1, label=r'$0.01$')
    ax1.plot(omega_axis, dataset2, color='k', linewidth=1.5,
             linestyle='-.', alpha=1, label=r'$0.03$')
    ax1.plot(omega_axis, dataset3, color='k', linewidth=1.5,
             linestyle='--', alpha=1, label=r'$0.06$')
    ax1.plot(omega_axis, stats.norm.pdf(omega, 0, 1), color='k',
             linewidth=1, linestyle=':', alpha=0.6, label=r'Gaussian')
    ax1.set_yscale('log')
    ax1.set_xlim((-np.pi, np.pi))
    ax1.set_ylabel(r'$f(x;t)$', fontsize=12, rotation=0, labelpad=23)
    ax1.set_xlabel(r'$x$', fontsize=12)

    xloglogs1 = omega**2 + 0.1564**2
    xloglogs2 = omega**2 + 0.4750**2
    xloglogs3 = omega**2 + 1.769**2

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_tick_params(labelright=True, labelleft=False)
    ax2.yaxis.tick_right()

    ax2.loglog(xloglogs1, dataset1, color='k', linestyle='-',
               linewidth=1.5,  alpha=1, label=r'$0.01$')
    ax2.loglog(xloglogs2, dataset2, color='k', linestyle='-.',
               linewidth=1.5,  alpha=1, label=r'$0.03$')
    ax2.loglog(xloglogs3, dataset3, color='k', linestyle='--',
               dashes=[6, 3, 6, 3, 6, 3], linewidth=1.5,  alpha=1, label=r'$0.06$')
    ax2.loglog(omega**2+1, stats.cauchy.pdf(omega, 0, 1), color='k', linewidth=1,
               linestyle=':', alpha=0.6, label=r'Cauchy, $\gamma=1$')
    ax2.set_ylabel(r'$f(x;t)$', fontsize=12, rotation=0, labelpad=23)
    ax2.set_xlabel(r'$x^2+\gamma^2$', fontsize=12)
    ax2.set_ylim((0.005, 4))

    legend1 = ax1.legend(
        title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False, loc='upper right')
    legend2 = ax2.legend(
        title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False, loc='upper right')
    plt.setp(legend1.get_title(), fontsize=16)
    plt.setp(legend2.get_title(), fontsize=16)

    plt.show()
#    fig.savefig('cont_mean_field_linlog_loglog.pdf')
    
    
    
"""
Fitting Wrapped Cauchy distribution to the data. Should improve fig 5.8 from the thesis for the paper
"""

def WrappedCauchy(x,gamma):             # no multiplication parameter A as in eqn 5.11 since these functions integrate to 1 on [-pi,pi] in stead of on [-inf,inf]
    x_0=0                                   # since all distributions are centered around 0
    return 1/(2*np.pi) * np.sinh(gamma) / (np.cosh(gamma) - np.cos(x-x_0))

if False:
    omega_axis = [val*radians for val in omega]
    dataset1 = data['0.01_1.0_1.0']['states_t%100'][-1]
    dataset2 = data['0.03_1.0_1.0']['states_t%100'][-1]
    dataset3 = data['0.06_1.0_1.0']['states_t%100'][-1]

    fitted_model1_WC_parms, fitted_model1_WC_cov = curve_fit(WrappedCauchy, omega, dataset1)
    fitted_model2_WC_parms, fitted_model2_WC_cov = curve_fit(WrappedCauchy, omega, dataset2) #      bounds=(0.068, [100, 0.075]))
    fitted_model3_WC_parms, fitted_model3_WC_cov = curve_fit(WrappedCauchy, omega, dataset3)
    
    
    fitted_model1_WC = WrappedCauchy(omega, *fitted_model1_WC_parms)
    fitted_model2_WC = WrappedCauchy(omega, *fitted_model2_WC_parms)
    fitted_model3_WC = WrappedCauchy(omega, *fitted_model3_WC_parms)

    fig = plt.figure(figsize=(1.75*6.4/2, 4.8))
    palette = plt.get_cmap('tab10')
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$f\, (x;t)$', fontsize=12, rotation=0, labelpad=20)
    ax1.set_xlim((-np.pi, np.pi))
    ax1.set_ylim((0, 2.2))
    ax1.plot(omega_axis, dataset1, color='k', linewidth=1,
             linestyle='-', alpha=0.25, label=r'$0.01$')
    ax1.plot(omega_axis, fitted_model1_WC, color='k',
             linewidth=1, linestyle='--', alpha=1, label=r'$0.01$ LSQ WC')
    ax1.plot(omega_axis, dataset2, color='k', linewidth=1.5,
             linestyle='-', alpha=0.25, label=r'$0.03$')
    ax1.plot(omega_axis, fitted_model2_WC, color='k',
             linewidth=1, linestyle='-.', alpha=1, label=r'$0.03$ LSQ WC')
    ax1.plot(omega_axis, dataset3, color='k', linewidth=2,
             linestyle='-', alpha=0.25, label=r'$0.06$')
    ax1.plot(omega_axis, fitted_model3_WC, color='k',
             linewidth=1.5, linestyle=':', alpha=1, label=r'$0.06$ LSQ WC')
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

    legend1 = ax1.legend(title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False)

    plt.setp(legend1.get_title(), fontsize=16)
    plt.show()
 #   fig.savefig('cont_mean_field_WrappedCauchy_LSQ_x0is0_BW.pdf')
    fitted_model1_WC_r2 = r2_score(dataset1, fitted_model1_WC)
    fitted_model2_WC_r2 = r2_score(dataset2, fitted_model2_WC)
    fitted_model3_WC_r2 = r2_score(dataset3, fitted_model3_WC)

    print(fitted_model1_WC_parms, 'r^2 = ', fitted_model1_WC_r2)
    print(fitted_model2_WC_parms, 'r^2 = ', fitted_model2_WC_r2)
    print(fitted_model3_WC_parms, 'r^2 = ', fitted_model3_WC_r2)
    


"""
Plotting wrapped Cauchy distr. fit parameter gamma vs system parameter ratio eta / (sigma k).
"""
def logfunc(x, mult):
    #this is a natural logarithm
    xoff = 0.068 +1e-10
    return  mult * (np.log(-x+xoff) - np.log(xoff-1e-10))

def polynom(x, alpha):
    x0=0.068+1e-10
    return 1/((-x+x0)**alpha) - 1/((x0)**alpha)


if True:
    gammas = np.zeros(len(data))
    etas = np.zeros(len(data))
    r2s = np.zeros(len(data))
    eps=1e-10
    i=0
    for run in data.keys():
        eta = data[run]['system']['n']
        if eta > 0.068-eps: # left bound of citical interval
            etas = etas[0:i]
            gammas = gammas[0:i]
            r2s = r2s[0:i]
            break
        states = data[run]['states_t%100'][-1]
        fitted_model_WC_parms, fitted_model_WC_cov = curve_fit(WrappedCauchy, omega, states, maxfev=500, p0=[1])
        fitted_model_WC = WrappedCauchy(omega, *fitted_model_WC_parms)
        fitted_model_WC_r2 = r2_score(states, fitted_model_WC)
        
       # print(fitted_model_WC_parms)
       # print(fitted_model_WC_r2)
        
        gammas[i] = fitted_model_WC_parms
        etas[i] = data[run]['system']['n']
        r2s[i] = fitted_model_WC_r2
        i+=1
        
        # if i==80:
        #     omega_axis = [val*radians for val in omega]

        #     fig2 = plt.figure()
        #     ax3 = fig2.add_subplot(1,1,1)
        #     ax3.set_xlim((-np.pi, np.pi))
        #     ax3.set_ylim((0, 2.2))
        #     ax3.plot(omega_axis, states, 'r', '-', label='data')
        #     ax3.plot(omega_axis, fitted_model_WC, 'b', '--', label='fit')
        #     plt.show()
    
    
    fitted_log_to_gamma_parms, fitted_log_to_gamma_cov = curve_fit(logfunc, etas, gammas, p0=[-0.6])
    fitted_log_to_gamma = logfunc(etas, *fitted_log_to_gamma_parms)
    fitted_log_to_gamma_r2 = r2_score(gammas, fitted_log_to_gamma)
    
    fitted_polynom_to_gamma_parms, fitted_log_to_gamma_cov = curve_fit(polynom, etas, gammas, p0=[10])
    fitted_polynom_to_gamma = polynom(etas, *fitted_polynom_to_gamma_parms)
    # fitted_polynom_to_gamma = polynom(etas, 1,-4,-1)
    fitted_polynom_to_gamma_r2 = r2_score(gammas, fitted_polynom_to_gamma)
    
    fig = plt.figure(figsize=(1.75*6.4*3/2, 4.82*1.5))
    palette = plt.get_cmap('tab10')
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.set_xlabel(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', fontsize=16)
    ax1.set_ylabel(r'$\gamma$', fontsize=14, rotation=0, labelpad=10)
    ax1.set_ylim((0, 3))
    ax1.set_xlim((0, 0.08))
    ax1.axvline(x=0.068, color='k', linestyle='--', alpha=0.2)
    ax1.axvline(x=0.072, color='k', linestyle='--', alpha=0.2)
    ax1.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.068$',
                 xy=(0.054, 0.3), xytext=(0.054, 0.3))
    ax1.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.072$',
                 xy=(0.073, 0.3), xytext=(0.073, 0.3))
    ax1.annotate(r'$\gamma = m\ \left(\log\left( -\frac{\eta}{\sigma_c \langle k \rangle^2} +0.068 \right) - \log(0.068) \right)$', xy=(0.005,2), xytext=(0.005,2))
    ax1.plot(etas, gammas, marker='x', color='k', linestyle = '')
    
    ax1.plot(etas, fitted_log_to_gamma, color='k', label='fit', alpha = 0.5)
   
    
    ax2.set_xlabel(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', fontsize=16)
    ax2.set_ylabel(r'$\gamma$', fontsize=14, rotation=0, labelpad=10)
    ax2.set_ylim((0, 3))
    ax2.set_xlim((0, 0.08))
    ax2.axvline(x=0.068, color='k', linestyle='--', alpha=0.2)
    ax2.axvline(x=0.072, color='k', linestyle='--', alpha=0.2)
    ax2.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.068$',
                 xy=(0.054, 0.3), xytext=(0.054, 0.3))
    ax2.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.072$',
                 xy=(0.073, 0.3), xytext=(0.073, 0.3))
    ax2.plot(etas, gammas, marker='x', color='k', linestyle = '')
    ax2.annotate(r'$\gamma = \frac{1}{(-\frac{\eta}{\sigma_c \langle k \rangle^2}+0.068)^\alpha} - \frac{1}{(0.068)^\alpha}$', xy=(0.02,2), xytext=(0.02,2))
    
    ax2.plot(etas, fitted_polynom_to_gamma, color='k', label='fit', alpha = 0.5)
    
    
    ax3.plot(etas,r2s, marker='x', color='k', linestyle='')
    ax3.axvline(x=0.068, color='k', linestyle='--', alpha=0.2)
    ax3.axvline(x=0.072, color='k', linestyle='--', alpha=0.2)
    ax3.set_xlabel(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', fontsize=16)
    ax3.set_ylabel(r'$R^2$', fontsize=14, rotation=0, labelpad=10)
    ax3.set_ylim((0.9,1))
    
    
    plt.show()
    # fig.savefig('wrapped_Cauchy_gamma_vs_eta_logfit_polyfit_r2.pdf')



"""
Plotting variance versus time and fitting an exponential decay function for paper
"""
def decay(t,a,tau, b):
    return a*np.exp(-t/tau) + b

if False:
    # we have to import another data set which contains many more time points
    palette = plt.get_cmap('tab10')
    filepath = str('var_vs_t/parameters.txt')
    paramfile = open(filepath, 'r')
    data_many_t = {}
    for line in paramfile:
        line1 = line.replace('\n', '')
        line2 = line1.replace(" ", "_")
        try:
            f = open("var_vs_t/data_manytpoints_{}_{}.dat".format(str(0), line2), "rb")
            data_many_t[line2] = pickle.load(f)
        except FileNotFoundError:
            print("var_vs_t/data_manytpoints_{}_{}.dat".format(str(0), line2), ' NOT FOUND')
    paramfile.close()

    keys = (['0.1_1.0_1.0', '0.08_1.0_1.0', '0.07_1.0_1.0', '0.06_1.0_1.0',
             '0.04_1.0_1.0', '0.03_1.0_1.0', '0.01_1.0_1.0'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(y=np.pi**2/3, color='k', linestyle='--', alpha=0.2)
    plt.annotate(r'$\sigma_f^2 = \frac{\pi^2}{3}$', xy=(
        10, np.pi**2/3-0.17), xytext=(10, np.pi**2/3-0.17))
    color = 0
    widths = [1.15, 1.15, 1.15, 1.15, 2, 2, 2, 2, 2]
    alphas = np.ones(7)
    styles = ['-', '--', '-.', ':', '-', '--', '-.']
    for key in keys:
        states = data_many_t[key]['states']
        var = np.zeros((states.shape[0]))               # 400 time steps are saved, out of 400, tmax = 400      
        for t in range((states.shape[0])):
            var[t] = variance(states[t, :], omega)
        lab = data_many_t[key]['system']['n']
        if lab == 0.1:
            lab = '0.10'
        if color < 4:
            ax.plot(range(300), var[0:300], 'k', linestyle=styles[color],
                    linewidth=widths[color], alpha=alphas[color], label=lab)  # color = palette(color)
        else:
            ax.plot(range(235), var[0:235], 'k', linestyle=styles[color],
                    linewidth=widths[color], alpha=alphas[color], label=lab)  # color = palette(color)
        
        variance_fit_parms, variance_fit_cov = curve_fit(decay, np.linspace(0,299,300), var[0:300])
        
        print(lab, variance_fit_parms[1]/33.333)
        
        if color < 4:
            length=300
        else:
            length=235
        ax.plot(np.linspace(0,length-1,length), decay(np.linspace(0,length-1,length), *variance_fit_parms), color='r',alpha=0.6)
        
        
        color += 1

       
    legend = plt.legend(loc=4, title=r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', frameon=False)
    plt.setp(legend.get_title(), fontsize=16)
    ax.set_ylim((0, 3.5))
    ax.set_xlim((0, 300))
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel(r'$\sigma_f^2$', fontsize=14, rotation=0, labelpad=15)
    plt.show()
#    fig.savefig('cont_mean_field_var_vs_t_exp_fits.pdf')


"""
Creating a plot for time constant versus ratio eta/sigma k for paper
"""
if False:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
 
    ax1.set_xlabel(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}$', fontsize=16)
    # labelpad increases distance label to axis
    ax1.set_ylabel(r'$\tau$', fontsize=14, rotation=0, labelpad=10)
#    ax1.set_ylim((0, 6))
    ax1.set_xlim((0, 0.15))

    # specify number of digits in ticks
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
   
    xdata = np.zeros(len(data.keys()))
    ydata = np.zeros(len(data.keys()))
    i = 0
    for run in data.keys():
        states = data[run]['states_t%100']              # 30 time steps are saved, out of (?300 000?), tmax = 1000. Time steps are 33.3x bigger than in the previous plot      
        n = data[run]['system']['n']
        var = np.zeros((states.shape[0]))
        for t in range((states.shape[0])):
            var[t] = variance(states[t, :], omega)
        variance_fit_parms, variance_fit_cov = curve_fit(decay, np.linspace(0,len(var)-1,len(var)), var)
        
        xdata[i] = n
        ydata[i] = variance_fit_parms[1]
        i += 1

    ax1.plot(xdata, ydata, 'k.', alpha=0.8, markersize=8)
    ax1.axvline(x=0.068, color='k', linestyle='--', alpha=0.2)
    ax1.axvline(x=0.072, color='k', linestyle='--', alpha=0.2)
    ax1.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.068$',
                 xy=(0.04, 2.7), xytext=(0.04, 2.7))
    ax1.annotate(r'$\frac{\eta}{\sigma_c \langle k \rangle^2}=0.072$',
                 xy=(0.077, 2.7), xytext=(0.077, 2.7))

    ax1.minorticks_on()
    plt.show()
#    fig.savefig('cont_mean_field_ratio_vs_timeconstant.pdf')
