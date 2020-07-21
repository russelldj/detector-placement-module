#!/usr/bin/env python
# coding: utf-8
from SDOptimizer.constants import DATA_FILE, PLOT_TITLES, ALARM_THRESHOLD, PAPER_READY, INFEASIBLE_MULTIPLE, NEVER_ALARMED_MULTIPLE, SMOOTH_PLOTS, INTERPOLATION_METHOD
from SDOptimizer.functions import make_location_objective, make_counting_objective, make_lookup, make_total_lookup_function, convert_to_spherical_from_points
from SDOptimizer.visualization import show_optimization_statistics, show_optimization_runs
from time import sleep
# from tqdm.notebook import trange, tqdm  # For plotting progress
from tqdm import trange, tqdm
from platypus import NSGAII, Problem, Real, Binary, Integer, CompoundOperator, SBX, HUX, PM, BitFlip
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from scipy.optimize import minimize, differential_evolution, rosen, rosen_der, fmin_l_bfgs_b
import os
import glob
import logging
import pdb
import scipy
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm

from importlib import reload
import io
import pandas as pd
import numpy as np
import matplotlib
import warnings
import copy
matplotlib.use('module://ipykernel.pylab.backend_inline')


class SDOptimizer():
    def __init__(self, interpolation_method=INTERPOLATION_METHOD, **kwargs):
        self.logger = logging.getLogger('main')
        self.logger.debug("Instantiated the optimizer")
        self.is3d = False
        self.X = None
        self.Y = None
        self.Z = None
        self.time_to_alarm = None

        self.interpolation_method = interpolation_method

    def visualize(self, show=False, log=True):
        """
        TODO update this so it outputs a video
        show : Boolean
            Do a matplotlib plot of every frame
        log : Boolean
            plot the concentration on a log scale
        """
        max_concentration = max(self.max_concentrations)

        print("Writing output files to ./vis")
        for i, concentration in tqdm(
            enumerate(
                self.concentrations), total=len(
                self.concentrations)):  # this is just wrapping it in a progress bar
            plt.cla()
            plt.clf()
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.title("concentration at timestep {} versus position".format(i))
            norm = mpl.colors.Normalize(vmin=0, vmax=1.0)

            cb = self.pmesh_plot(
                self.X,
                self.Y,
                concentration,
                plt,
                log=log,
                max_val=max_concentration)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.savefig("vis/concentration{:03d}.png".format(i))
            if show:
                plt.show()

    def get_3D_locs(self):
        return (self.X, self.Y, self.Z)

    def example_time_to_alarm(self, x_bounds, y_bounds,
                              center, show=False, scale=1, offset=0):
        """
        Xs and Ys are the upper and lower bounds
        center is the x y coords
        scale is a multiplicative factor
        offset is additive
        """
        Xs = np.linspace(*x_bounds)
        Ys = np.linspace(*y_bounds)
        x, y = np.meshgrid(Xs, Ys)
        z = (x - center[0]) ** 2 + (y - center[1]) ** 2
        z = z * scale + offset
        if show:
            plt.cla()
            plt.clf()
            cb = plt.pcolormesh(x, y, z, cmap=plt.cm.inferno)
            plt.colorbar(cb)  # Add a colorbar to a plot
            plt.pause(4.0)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        return (x, y, z)

    def make_platypus_objective_function(
            self, sources, func_type="basic", bad_sources=[]):
        """
        sources

        bad_sources : ArrayLike[Sources]
            These are the ones which we don't want to be near
        """
        if func_type == "basic":
            raise NotImplementedError("I'm not sure I'll ever do this")
            # return self.make_platypus_objective_function_basic(sources)
        elif func_type == "counting":
            return self.make_platypus_objective_function_counting(sources)
        elif func_type == "competing_function":
            return self.make_platypus_objective_function_competing_function(
                sources, bad_sources)
        else:
            raise ValueError("The type : {} is not an option".format(func_type))

    def make_platypus_objective_function_competing_function(
            self, sources, bad_sources=[]):
        total_ret_func = make_total_lookup_function(
            sources, interpolation_method=self.interpolation_method)  # the function to be optimized
        bad_sources_func = make_total_lookup_function(
            bad_sources, type="fastest",
            interpolation_method=self.interpolation_method)  # the function to be optimized

        def multiobjective_func(x):  # this is the double objective function
            return [total_ret_func(x), bad_sources_func(x)]

        num_inputs = len(sources) * 2  # there is an x, y for each source
        NUM_OUPUTS = 2  # the default for now
        # define the demensionality of input and output spaces
        problem = Problem(num_inputs, NUM_OUPUTS)
        x, y, time = sources[0]  # expand the first source
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        print(
            "min x : {}, max x : {}, min y : {}, max y : {}".format(
                min_x,
                max_x,
                min_y,
                max_y))
        problem.types[::2] = Real(min_x, max_x)  # This is the feasible region
        problem.types[1::2] = Real(min_y, max_y)
        problem.function = multiobjective_func
        # the second function should be maximized rather than minimized
        problem.directions[1] = Problem.MAXIMIZE
        return problem

    def make_platypus_objective_function_counting(
            self, sources, times_more_detectors=1):
        """
        This balances the number of detectors with the quality of the outcome
        """
        total_ret_func = make_total_lookup_function(
            sources, masked=True)  # the function to be optimized
        counting_func = make_counting_objective()

        def multiobjective_func(x):  # this is the double objective function
            return [total_ret_func(x), counting_func(x)]

        # there is an x, y, and a mask for each source so there must be three
        # times more input variables
        # the upper bound on the number of detectors n times the number of
        # sources
        num_inputs = len(sources) * 3 * times_more_detectors
        NUM_OUPUTS = 2  # the default for now
        # define the demensionality of input and output spaces
        problem = Problem(num_inputs, NUM_OUPUTS)
        x, y, time = sources[0]  # expand the first source
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        print(
            "min x : {}, max x : {}, min y : {}, max y : {}".format(
                min_x,
                max_x,
                min_y,
                max_y))
        problem.types[0::3] = Real(min_x, max_x)  # This is the feasible region
        problem.types[1::3] = Real(min_y, max_y)
        # This appears to be inclusive, so this is really just (0, 1)
        problem.types[2::3] = Binary(1)
        problem.function = multiobjective_func
        return problem

    def plot_inputs(self, inputs, optimized, show_optimal=False):
        plt.cla()
        plt.clf()
        f, ax = self.get_square_axis(len(inputs))
        max_z = 0
        for i, (x, y, z) in enumerate(inputs):
            max_z = max(max_z, max(z))  # record this for later plotting
            cb = self.pmesh_plot(x, y, z, ax[i])
            if show_optimal:
                for j in range(0, len(optimized), 2):
                    detectors = ax[i].scatter(optimized[j], optimized[j + 1],
                                              c='w', edgecolors='k')
                    ax[i].legend([detectors], ["optimized detectors"])
        f.colorbar(cb)
        if PAPER_READY:
            plt.savefig("vis/TimeToAlarmComposite.png")
        f.suptitle("The time to alarm for each of the smoke sources")
        plt.show()
        return max_z

    def get_square_axis(self, num, is_3d=False):
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

    def plot_sweep(self, xytimes, fixed_detectors,
                   bounds, max_val=None, centers=None):
        """
        xytimes : ArrayLike[Tuple[]]
            the smoke propagation information
        xs : ArrayLike
            [x1, y1, x2, y2...] representing the fixed location of the smoke detectors
        bounds : ArrayLike
            [x_low, x_high, y_low, y_high] the bounds on the swept variable
        """
        # TODO refactor so this is the same as the other one
        time_func = make_total_lookup_function(xytimes)
        print(time_func)
        x_low, x_high, y_low, y_high = bounds
        xs = np.linspace(x_low, x_high)
        ys = np.linspace(y_low, y_high)
        grid_xs, grid_ys = np.meshgrid(xs, ys)
        grid_xs = grid_xs.flatten()
        grid_ys = grid_ys.flatten()
        grid = np.vstack((grid_xs, grid_ys)).transpose()
        print(grid.shape)
        times = []
        for xy in grid:
            locations = np.hstack((fixed_detectors, xy))
            times.append(time_func(locations))
        plt.cla()
        plt.clf()

        cb = self.pmesh_plot(grid_xs, grid_ys, times, plt, max_val)
        # even and odd points
        fixed = plt.scatter(fixed_detectors[::2], fixed_detectors[1::2], c='k')
        plt.colorbar(cb)  # Add a colorbar to a plot
        if centers is not None:
            centers = plt.scatter(centers[::2], centers[1::2], c='w')
            plt.legend([fixed, centers], [
                       "The fixed detectors", "Centers of smoke sources"])
        else:
            plt.legend([fixed], ["The fixed detectors"])
        plt.title("Effects of placing the last detector with {} fixed".format(
            int(len(fixed_detectors) / 2)))
        plt.show()

    def plot_3d(
            self,
            xs,
            ys,
            values,
            plotter,
            max_val=None,
            num_samples=50,
            is_3d=False,
            cmap=plt.cm.inferno):
        """
        conveneince function to easily plot the sort of data we have
        """
        points = np.stack((xs, ys), axis=1)
        sample_points = (np.linspace(min(xs), max(xs), num_samples),
                         np.linspace(min(ys), max(ys), num_samples))
        xis, yis = np.meshgrid(*sample_points)
        flattened_xis = xis.flatten()
        flattened_yis = yis.flatten()
        interpolated = griddata(points, values, (flattened_xis, flattened_yis))
        reshaped_interpolated = np.reshape(interpolated, xis.shape)
        if max_val is not None:
            norm = mpl.colors.Normalize(0, max_val)
        else:
            norm = mpl.colors.Normalize()  # default

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        cb = ax.plot_surface(
            xis,
            yis,
            reshaped_interpolated,
            cmap=cmap,
            norm=norm,
            edgecolor='none')
        ax.set_title('Surface plot')
        plt.show()
        return cb  # return the colorbar

    def visualize_all(
            self,
            objective_func,
            optimized_detectors,
            bounds,
            max_val=None,
            num_samples=30,
            verbose=False,
            is3d=False,
            log=False):
        """
        The goal is to do a sweep with each of the detectors leaving the others fixed
        """
        # set up the sampling locations
        x_low, x_high, y_low, y_high = bounds
        xs = np.linspace(x_low, x_high, num_samples)
        ys = np.linspace(y_low, y_high, num_samples)
        grid_xs, grid_ys = np.meshgrid(xs, ys)
        grid_xs = grid_xs.flatten()
        grid_ys = grid_ys.flatten()
        # This is a (n, 2) where each row is a point
        grid = np.vstack((grid_xs, grid_ys)).transpose()

        # create the subplots
        plt.cla()
        plt.clf()
        # f, ax = plt.subplots(int(len(optimized_detectors)/2), 1)
        f, ax = self.get_square_axis(len(optimized_detectors) / 2)

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

            cb = self.pmesh_plot(
                grid_xs,
                grid_ys,
                times,
                which_plot,
                max_val,
                log=log)

            fixed = which_plot.scatter(
                selected_detectors[::2], selected_detectors[1::2], c='w', edgecolors='k')

            if verbose:
                which_plot.legend([fixed], ["the fixed detectors"])
                which_plot.set_xlabel("x location")
                which_plot.set_ylabel("y location")

        plt.colorbar(cb, ax=ax[-1])
        if PAPER_READY:
            # write out the number of sources
            plt.savefig(
                "vis/DetectorSweeps{:02d}Sources.png".format(
                    int(len(optimized_detectors) / 2)))
        f.suptitle("The effects of sweeping one detector with all other fixed")

        plt.show()

    def evaluate_optimization(self, sources, num_detectors, bounds=None,
                              genetic=True, visualize=True, num_iterations=10):
        """
        sources : ArrayLike
            list of (x, y, time) tuples
        num_detectors : int
            The number of detectors to place
        bounds : ArrayLike
            [x_low, x_high, y_low, y_high], will be computed from self.X, self.Y if None
        genetic : Boolean
            whether to use a genetic algorithm
        visualize : Boolean
            Whether to visualize the results
        num_iterations : int
            How many times to run the optimizer
        """
        vals = []
        locs = []
        iterations = []
        func_values = []
        for i in trange(num_iterations):
            res = self.optimize(
                sources,
                num_detectors,
                bounds=bounds,
                genetic=genetic,
                visualize=False)
            vals.append(res.fun)
            locs.append(res.x)
            iterations.append(res.nit)
            func_values.append(res.vals)

        if visualize:
            show_optimization_statistics(vals, iterations, locs)
            show_optimization_runs(func_values)

        return vals, locs, iterations, func_values

    def show_optimization_statistics(self, vals, iterations, locs):
        show_optimization_statistics(vals, iterations, locs)

    def set_3d(self, value=False):
        """
        set whether it should be 3d
        """
        self.is3d = value

    def test_tqdm(self):
        for _ in trange(30):  # For plotting progress
            sleep(0.5)


if __name__ == "__main__":  # Only run if this was run from the commmand line
    SDO = SDOptimizer()
    SDO.load_data(DATA_FILE)  # Load the data file
    X1, Y1, time_to_alarm1 = SDO.get_time_to_alarm(False)
    X2, Y2, time_to_alarm2 = SDO.example_time_to_alarm(
        (0, 1), (0, 1), (0.3, 0.7), False)
    ret_func = make_lookup(X1, Y1, time_to_alarm1)
    total_ret_func = make_total_lookup_function(
        [(X1, Y1, time_to_alarm1), (X2, Y2, time_to_alarm2)])

    CENTERS = [0.2, 0.8, 0.8, 0.8, 0.8, 0.2]

    x1, y1, z1 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[0:2], False)
    x2, y2, z2 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[2:4], False)
    x3, y3, z3 = SDO.example_time_to_alarm([0, 1], [0, 1], CENTERS[4:6], False)
    inputs = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

    total_ret_func = make_total_lookup_function(inputs)
    BOUNDS = ((0, 1), (0, 1), (0, 1), (0, 1))  # constraints on inputs
    INIT = (0.51, 0.52, 0.47, 0.6, 0.55, 0.67)
    res = minimize(total_ret_func, INIT, method='COBYLA')
    print(res)
    x = res.x
