import pdb
import warnings
import logging
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
import pyvista as pv

from .constants import (PAPER_READY, PLOT_TITLES, SMOOTH_PLOTS)
from .functions import normalize


# matplotlib.use('module://ipykernel.pylab.backend_inline')


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


def visualize_3D_with_final(smoke_source, final_locations=None,
                            label="3D visualization of the time to alarm",
                            plotter=None):
    """
    XYZ : np.array, (n, 3)
        The 3D locations of the points
    smoke_source : dict
        The coresponding result from `get_time_to_alarm()``
    final_locations : [(x, y), (x, y), ...]
        The location(s) of the detector placements
    show : bool
        whether to show
    plotter : pv.Plotter
        existing plotter to use
    """
    warnings.warn("Untested: may give spurious results.")
    # TODO update this to accomodate the new smoke sources
    plotter = visualize_3D(smoke_source.parameterized_locations,
                           smoke_source.time_to_alarm,
                           label=label, plotter=plotter, show=False)

    # These parameterize the space we optimizaed over
    # All of these X, Y, Z, x, y, z, time_to_alarm should be the same
    # length with corresponding indices
    parameterized_locations = smoke_source.parameterized_locations
    dimensionality = parameterized_locations.shape[1]

    XYZ = smoke_source.parameterized_locations

    for i in range(0, len(final_locations), dimensionality):
        final_location = final_locations[i:i+dimensionality]
        # Find the index of the nearest point
        diffs = parameterized_locations - final_location
        dists = np.linalg.norm(diffs, axis=1)
        min_loc = np.argmin(dists)
        closest_point = XYZ[min_loc, :]
        highlight = pv.Sphere(radius=0.15, center=closest_point)
        plotter.add_mesh(highlight, color="red")

    plotter.show()


def visualize_3D(XYZ, time_to_alarm,
                 label="3D visualization of the time to alarm",
                 plotter=None, show=True):
    """
    XYZ_locs : (X, Y, Z)
        The 3D locations of the points
    time_to_alarm : ArrayLike[Floats]
        How long it takes for each point to alarm
    fraction : float
        how much of the points to visualize
    plotter : pv.Plotter | None
        Can plot on existing axis
    show : bool
        Don't show so more information can be added
    """
    if plotter is None:
        plotter = pv.Plotter()
    mesh = pv.PolyData(XYZ)
    # This will colormap the values
    plotter.add_mesh(mesh, scalars=time_to_alarm,  stitle='Time to alarm')
    # Don't show so other data can be added easily
    if show:
        plotter.show(screenshot="vis.png")
    return plotter


def visualize_time_to_alarm(parameterized_locations, time_to_alarm, num_samples,
                            concentrations, num_samples_visualized=10,
                            smoothed=SMOOTH_PLOTS, spherical=True,
                            write_figs=PAPER_READY,
                            axis_labels=("x location", "y location")):
    """
    show the time to alarm plots

    paramerterized_locations : np.array
        n samples x m parameterizing variables
    """
    parameterizing_dimensionality = parameterized_locations.shape[1]
    if parameterizing_dimensionality == 2:

        cb = pmesh_plot(
            parameterized_locations[:, 0],
            parameterized_locations[:, 1],
            time_to_alarm,
            plt,
            num_samples=70, smooth=smoothed,
            cmap=mpl.cm.inferno)  # choose grey to plot color over

        plt.colorbar(cb)  # Add a colorbar to a plot
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        if write_figs:
            if smoothed:
                plt.savefig("vis/TimeToAlarmSmoothed.png")
            else:
                plt.savefig("vis/TimeToAlarmDots.png")
        plt.show()
    elif parameterizing_dimensionality == 3:
        visualize_3D(parameterized_locations, time_to_alarm)
    else:
        logging.error(
            "visualize_time_to_alarm only supports data with 2 or 3 dimensions but {parameterizing_dimensionality} were given")


def visualize_additional_time_to_alarm_info(X, Y, Z, time_to_alarm,
                                            num_samples, concentrations,
                                            num_samples_visualized=10,
                                            smoothed=SMOOTH_PLOTS,
                                            write_figs=PAPER_READY,
                                            axis_labels=("x location",
                                                         "y location")):
    cb = pmesh_plot(
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
    # cb = plt.scatter(xs.flatten(), ys.flatten(), c=concentrations.flatten())
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
        if is_3d:
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
                xis, yis, reshaped_interpolated, cmap=cmap, norm=norm,
                shading="nearest")
    else:  # Not smooth
        # Just do a normal scatter plot
        cb = plotter.scatter(xs, ys, c=values, cmap=cmap)
    return cb  # return the colorbar


def visualize_slices(
        objective_func,
        optimized_detectors,
        bounds,
        max_val=None,
        num_samples=30,
        is_3d=False,
        log=False,
        axis_labels=("phi", "theta")):
    """
    The goal is to do a sweep with each of the detectors leaving the others fixed

    bounds [(x_low, x_high), (y_low, y_high), ....]
    """
    if is_3d:
        logging.error("Cannot visualize slices for 3D parameterization")
        return None

    logging.warn("Begining to visualize slices. May take a while")

    # set up the sampling locations

    (x_low, x_high), (y_low, y_high) = bounds[:2]
    xs = np.linspace(x_low, x_high, num_samples)
    ys = np.linspace(y_low, y_high, num_samples)
    grid_xs, grid_ys = np.meshgrid(xs, ys)
    grid_xs = grid_xs.flatten()
    grid_ys = grid_ys.flatten()
    # This is a (n, 2) where each row is a point
    grid = np.vstack((grid_xs, grid_ys)).transpose()

    # TODO get rid of is_3d
    # make this work with 3D
    f, ax = get_square_axis(len(optimized_detectors) / 2, is_3d=is_3d)

    num_samples = grid.shape[0]
    for i in range(0, len(optimized_detectors), 2):
        selected_detectors = np.concatenate(
            (optimized_detectors[:i], optimized_detectors[(i + 2):]), axis=0)  # get all but one

        repeated_selected = np.tile(np.expand_dims(
            selected_detectors, axis=0), reps=(num_samples, 1))
        locations = np.concatenate((grid, repeated_selected), axis=1)

        times = [objective_func(xys) for xys in locations]
        if isinstance(ax, np.ndarray):  # ax may be al
            which_plot = ax[int(i / 2)]
        else:
            which_plot = ax

        cb = pmesh_plot(
            grid_xs,
            grid_ys,
            times,
            which_plot,
            max_val,
            log=log)

        fixed = which_plot.scatter(
            selected_detectors[::2], selected_detectors[1::2], c='w', edgecolors='k')

        which_plot.legend([fixed], ["the fixed detectors"])
        which_plot.set_xlabel(axis_labels[0])
        which_plot.set_ylabel(axis_labels[1])

    plt.colorbar(cb, ax=ax[-1])
    if PAPER_READY:
        # write out the number of sources
        plt.savefig(
            "vis/DetectorSweeps{:02d}Sources.png".format(
                int(len(optimized_detectors) / 2)))
    f.suptitle("The effects of sweeping one detector with all other fixed")
    plt.show()


def visualize_sources(sources, final_locations):
    """
    sources : [dict, ...]
    final_locations : [x1, y1, x2, y2, ....]

    returns None
    """
    dimensionality = sources[0].parameterized_locations.shape[1]

    if dimensionality == 2:
        x_detector_inds = np.arange(0, len(final_locations), 2).astype(int)
        y_detector_inds = x_detector_inds + 1

        f, ax = get_square_axis(len(sources))
        for i, source in enumerate(sources):
            x = source.parameterized_locations[:, 0]
            y = source.parameterized_locations[:, 1]
            axis_labels = source.axis_labels
            time_to_alarm = source.time_to_alarm
            cb = pmesh_plot(x, y, time_to_alarm, ax[i])

            detectors = ax[i].scatter(final_locations[x_detector_inds],
                                      final_locations[y_detector_inds],
                                      c='g', edgecolors='w')
            ax[i].legend([detectors], ["optimized detectors"])

            ax[i].set_xlabel(axis_labels[0])
            ax[i].set_ylabel(axis_labels[1])

        f.colorbar(cb)
        if PAPER_READY:
            plt.savefig("vis/TimeToAlarmComposite.png")
        f.suptitle("The time to alarm for each of the smoke sources")
        plt.show()

    elif dimensionality == 3:

        for i, source in enumerate(sources):
            # record this for later plotting
            # TODO figure out if this is really required
            visualize_3D_with_final(source, final_locations=final_locations)
    else:
        logging.error(
            f"visualize_sources only supports 2 or 3 dimensions but recieved {dimensionality}")


def get_square_axis(num, is_3d=False):
    """
    arange subplots in a rough square based on the number of inputs
    """
    if num == 1:
        if is_3d:
            f, ax = plt.subplots(1, 1, projection='3d')
        else:
            f, ax = plt.subplots(1, 1)

        ax = np.asarray([ax])
        return f, ax
    num_x = np.ceil(np.sqrt(num))
    num_y = np.ceil(num / num_x)
    if is_3d:
        f, ax = plt.subplots(int(num_y), int(num_x), projection='3d')
    else:
        f, ax = plt.subplots(int(num_y), int(num_x))

    ax = ax.flatten()
    return f, ax
