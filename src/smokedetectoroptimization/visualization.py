import pdb
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from .constants import (PLOT_TITLES, BIG_NUMBER, ALARM_THRESHOLD,
                        PAPER_READY, SMOOTH_PLOTS)
from .functions import normalize
matplotlib.use('module://ipykernel.pylab.backend_inline')


def show_optimization_statistics(vals, iterations, locs):
    plt.hist(vals, bins=10)  # plot the bins as a quarter of the spread
    # ax1.violinplot(vals)
    plt.xlabel("objective function values")
    if PAPER_READY:
        plt.savefig("vis/SummaryOptimizationValues.png")
    plt.show()

    plt.hist(iterations, bins=10)
    # TODO make this a histogram
    plt.xlabel("number of evaluations to converge")
    if PAPER_READY:
        plt.savefig("vis/SummaryOptimizationIterations.png")
    plt.show()

    for loc in locs:
        plot_xy(loc)
    plt.xlim(0, 8.1)
    plt.ylim(0, 3)
    plt.show()


def show_optimization_runs(all_funcs_values):
    """
    show the distribution of training curves

    all_funcs_values : ArrayLike[ArrayLike]
        the objective function value versus iteration for all the runs
    """
    for func_values in all_funcs_values:
        plt.plot(func_values)
    plt.xlabel("Number of iterations")
    plt.ylabel("Objective function value")
    if PAPER_READY:
        plt.savefig("vis/ObjectiveFunctionValues.png")
    plt.title("The plot of all the objective functions for a set of runs")
    plt.show()

    # plot the summary statistics


def plot_xy(xy):
    plt.scatter(xy[::2], xy[1::2])


def plot_sphere(phi, theta, cs, r=1):
    phi = normalize(phi, -np.pi, 2 * np.pi)
    theta = normalize(theta, -np.pi, 2 * np.pi)
    xs = r * np.sin(phi) * np.cos(theta)
    ys = r * np.sin(phi) * np.sin(theta)
    zs = r * np.cos(phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    plt.savefig("sphere.png")
    pdb.set_trace()


def visualize_3D(XYZ_locs, smoke_source, final_locations,
                 label="3D visualization of the time to alarm",
                 fraction=0.05):
    """
    XYZ_locs : (X, Y, Z)
        The 3D locations of the points
    smoke_source : (x, y, time_to_alarm)
        The coresponding result from `get_time_to_alarm()``
    final_locations : [(x, y), (x, y), ...]
        The location(s) of the detector placements
    fraction : float
        how much of the points to visualize
    """
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # TODO see if there is a workaround to get equal aspect
    # Unpack
    X, Y, Z = XYZ_locs
    x, y, time_to_alarm = smoke_source
    xy = np.hstack((np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)))
    for i in range(0, len(final_locations), 2):
        final_location = final_locations[i:i+2]
        # Find the index of the nearest point
        diffs = xy - final_location
        dists = np.linalg.norm(diffs, axis=1)
        min_loc = np.argmin(dists)
        closest_X = X[min_loc]
        closest_Y = Y[min_loc]
        closest_Z = Z[min_loc]
        ax.scatter(closest_X, closest_Y, closest_Z,
                   s=200, c='chartreuse', linewidths=0)
    num_points = len(X)  # could be len(Y) or len(Z)
    sample_points = np.random.choice(num_points,
                                     size=(int(num_points * fraction),))
    cb = ax.scatter(X[sample_points], Y[sample_points], Z[sample_points],
                    c=time_to_alarm[sample_points], cmap=cm.inferno, linewidths=1)
    plt.colorbar(cb)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    print(label)
    ax.set_label(label)
    plt.show()


def visualize_time_to_alarm(self, X, Y, time_to_alarm, num_samples,
                            concentrations, num_samples_visualized=10,
                            smoothed=SMOOTH_PLOTS, spherical=True,
                            write_figs=PAPER_READY):
    cb = self.pmesh_plot(
        X,
        Y,
        time_to_alarm,
        plt,
        num_samples=70, smooth=smoothed,
        cmap=mpl.cm.inferno)  # choose grey to plot color over

    plt.colorbar(cb)  # Add a colorbar to a plot
    if spherical:
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\phi$')
    else:
        plt.xlabel("x location")
        plt.ylabel("y location")
    if write_figs:
        if smoothed:
            plt.savefig("vis/TimeToAlarmSmoothed.png")
        else:
            plt.savefig("vis/TimeToAlarmDots.png")
    plt.show()
    cb = self.pmesh_plot(
        X,
        Y,
        time_to_alarm,
        plt,
        num_samples=70,
        cmap=mpl.cm.Greys)  # choose grey to plot color over
    plt.colorbar(cb)  # Add a colorbar to a plot
    if PLOT_TITLES:
        plt.title("Time to alarm versus location on the wall")
    plt.xlabel("X location")
    plt.ylabel("Y location")
    samples = np.random.choice(
        num_samples,
        num_samples_visualized)
    # plot the sampled locations
    xs = X[samples]
    ys = Y[samples]
    for x_, y_ in zip(xs, ys):  # dashed to avoid confusion
        plt.scatter(x_, y_)
    if write_figs:
        plt.savefig("vis/SingleTimestepConcentration.png")
    plt.show()
    rows = concentrations[:, samples].transpose()
    for row in rows:
        plt.plot(row)
    if PLOT_TITLES:
        plt.title("random samples of concentration over time")
    plt.xlabel("timesteps")
    plt.ylabel("concentration")
    if write_figs:
        plt.savefig("vis/ConsentrationsOverTime.png")
    plt.show()

    # This is now the first concentrations
    last_concentrations = concentrations[0, :]
    nonzero_concentrations = last_concentrations[np.nonzero(
        last_concentrations)]
    log_nonzero_concentrations = np.log10(nonzero_concentrations)

    plt.hist(log_nonzero_concentrations)
    plt.xlabel("Final concentration (log)")
    plt.ylabel("Frequency of occurance")
    if write_figs:
        plt.savefig("vis/FinalStepConcentrationHist.png")
    plt.title(
        "The histogram of the final nonzero log_{10} smoke concentrations")
    plt.show()

    plt.hist(time_to_alarm)
    plt.xlabel("Time to alarm (timesteps)")
    plt.ylabel("Frequency of occurance")
    if write_figs:
        plt.savefig("vis/TimeToAlarmHistogram.png")
    plt.title(
        "The histogram of the time to alarm")
    plt.show()

    # show all the max_concentrations
    # This takes an extrodinarily long time
    # xs, ys = np.meshgrid(
    #    range(concentrations.shape[1]), range(concentrations.shape[0]))
    # pdb.set_trace()
    #cb = plt.scatter(xs.flatten(), ys.flatten(), c=concentrations.flatten())
    # plt.colorbar(cb)  # Add a colorbar to a plot
    # plt.show()


def pmesh_plot(
        xs,
        ys,
        values,
        plotter,
        max_val=None,
        num_samples=50,
        is_3d=False,
        log=False,  # log scale for plotting
        smooth=SMOOTH_PLOTS,
        cmap=plt.cm.inferno):
    """
    conveneince function to easily plot the sort of data we have
    smooth : Boolean
        Plot the interpolated values rather than the actual points

    """
    if smooth:
        points = np.stack((xs, ys), axis=1)
        sample_points = (np.linspace(min(xs), max(xs), num_samples),
                         np.linspace(min(ys), max(ys), num_samples))
        xis, yis = np.meshgrid(*sample_points)
        flattened_xis = xis.flatten()
        flattened_yis = yis.flatten()
        interpolated = griddata(
            points, values, (flattened_xis, flattened_yis))
        reshaped_interpolated = np.reshape(interpolated, xis.shape)
        if max_val is not None:
            if log:
                EPSILON = 0.0000000001
                norm = mpl.colors.LogNorm(
                    EPSILON, max_val + EPSILON)  # avoid zero values
                reshaped_interpolated += EPSILON
            else:
                norm = mpl.colors.Normalize(0, max_val)
        else:
            if log:
                norm = mpl.colors.LogNorm()
                EPSILON = 0.0000000001
                reshaped_interpolated += EPSILON
            else:
                norm = mpl.colors.Normalize()  # default

        # TODO see if this can be added for the non-smooth case
        if is3d:
            plt.cla()
            plt.clf()
            plt.close()
            ax = plt.axes(projection='3d')
            # cb = ax.plot_surface(xis, yis, reshaped_interpolated,cmap=cmap, norm=norm, edgecolor='none')
            cb = ax.contour3D(
                xis, yis, reshaped_interpolated, 60, cmap=cmap)
            plt.show()
        else:
            cb = plotter.pcolormesh(
                xis, yis, reshaped_interpolated, cmap=cmap, norm=norm)
    else:  # Not smooth
        # Just do a normal scatter plot
        cb = plotter.scatter(xs, ys, c=values, cmap=cmap)
    return cb  # return the colorbar
