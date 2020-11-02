# -*- coding: utf-8 -*-
"""
Most optimization functions should be accessible from this file
"""

import argparse
import sys
import logging
import pdb
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from .functions import make_objective_function
from .constants import (SINGLE_OBJECTIVE_FUNCTIONS, MULTI_OBJECTIVE_FUNCTIONS,
                        PAPER_READY, NUM_EVALUATION_ITERS)
from .visualization import (visualize_slices, visualize_sources,
                            visualize_3D_with_highlights,
                            show_optimization_statistics,
                            show_optimization_runs)

from smokedetectoroptimization import __version__


__author__ = "David Russell"
__copyright__ = "David Russell"
__license__ = "mit"

optimization_logger = logging.getLogger(__name__)


def evaluate_optimization(sources,
                          num_iterations=NUM_EVALUATION_ITERS,
                          visualize_summary=True,
                          **kwargs):
    """
    Use the same keywords as optimize

    kwargs is the keyword arguments
    """
    # Create a dictionary with a default value of the empty list
    statistics = defaultdict(lambda: [])

    for _ in trange(num_iterations):
        res = optimize(sources=sources, vis=False, **kwargs)
        statistics["iter_vals"].append(res.iter_vals)
        statistics["final_vals"].append(res.iter_vals[0])
        statistics["final_locs"].append(res.x)
        statistics["num_iters"].append(res.nit)
    if visualize_summary:
        show_optimization_statistics(statistics["final_vals"],
                                     statistics["num_iters"],
                                     statistics["final_locs"])
        show_optimization_runs(statistics["iter_vals"])
    return statistics


def evaluate_locations(locations,
             sources,
             function_type="worst_case_TTA",
             bad_sources=None,
             vis=True,
             interpolation_method="nearest",
             parameterized=False):
    """similar arguements to optimize. Except instead
    of doing optimization, it just reports the function value for the location

    locations : ArrayLike
        Concatenated detector locations to evaluate provided in XYZ space
    parameterized : bool
        Are the locations parameterized
    """
    if len(sources) == 0:
        raise ValueError("Empty sources")

    bounds = None
    NUM_DETECTORS = 3

    parameterized_XYZ_points = sources[0].get_closest_points(locations,
                                                             parameterized=parameterized)
    parameterized_points = [d["parameterized"] for
                            d in parameterized_XYZ_points]

    # Flatten the nested list of locations
    parameterized_points = sum([x.tolist() for x in parameterized_points], [])

    # TODO calculate the number of dectectors from the location
    optimization_logger.info("Making bounds")
    bounds = make_bounds(bounds, sources, NUM_DETECTORS)

    optimization_logger.info("Making the objective function")
    # compute the bounds
    objective_function = make_objective_function(
        sources=sources,
        bounds=bounds,
        function_type=function_type,
        bad_sources=bad_sources,
        interpolation_method=interpolation_method)
    objective_value = objective_function(parameterized_points)
    if vis:
        for source in sources:
            pass

    return objective_value

def optimize(sources,
             num_detectors=1,
             function_type="worst_case_TTA",
             bounds=None,
             bad_sources=None,
             vis=True,
             interpolation_method="nearest"):
    """
    sources : [SmokeSource]
        Sources are now represented by their own class
    num_detectors : int
        The number of detectors to place
    bounds : ArrayLike
        [x_low, x_high, y_low, y_high] or [
            x_low, x_high, y_low, y_high, z_low, z_high]
        will be computed from sources if None. This determines whether to
        optimize over a two or three dimensional space
    genetic : Boolean
        whether to use a genetic algorithm
    masked : Boolean
        Whether the input is masked TODO figure out what I meant
    multiobjective : Boolean
        Should really be called multiobjective. Runs multiobjective
    function_type : str
        What function to optimize : 'worst_case', TODO
    """
    optimization_logger.info("Making bounds")
    bounds = make_bounds(bounds, sources, num_detectors)

    optimization_logger.info("Making the objective function")
    # compute the bounds
    objective_function = make_objective_function(
        sources=sources,
        bounds=bounds,
        function_type=function_type,
        bad_sources=bad_sources,
        interpolation_method=interpolation_method)  # This is keyword arguments which are passed from the call to optimize

    if function_type in SINGLE_OBJECTIVE_FUNCTIONS:
        # Do the single objective optimization
        optimization_logger.info("Running single objective optimization")
        res = run_single_objective_problem(
            objective_function, bounds)

        if vis:
            visualize_single_objective_problem(objective_function,
                                               res.iter_vals,
                                               sources, bounds, res.x)
            sources[0].describe_closest_points(res.x)
            print(f"The final value was {res.iter_vals[-1]}")

        return res

    elif function_type in MULTI_OBJECTIVE_FUNCTIONS:
        optimization_logger.warning("Running mulitobjecitve optimization")
        run_multiobjective_problem(objective_function,
                                   function_type=function_type)


def run_single_objective_problem(objective_function, bounds):
    """
    Perform single objective optimization

    objective_function : func(ArrayLike) -> float
    bounds : bounds
    compute_function_values, bool : compute the values of the function at each iteration
    """

    locations_over_iterations = []

    def callback(xk, convergence):
        locations_over_iterations.append(xk)

    optimization_logger.warning("About to call scipy for optimzation")
    # This is where the actual optimization occurs
    res = differential_evolution(objective_function, bounds, callback=callback)
    optimization_logger.warning("Completed optimzation")

    optimization_logger.warning(
        f"Starting to compute values for plotting. May take a while.")
    iter_vals = [objective_function(
        x) for x in locations_over_iterations]
    optimization_logger.warning(f"Done computing values for plotting")

    # set a new field to record the function values over time
    res.iter_vals = iter_vals

    return res


def visualize_single_objective_problem(objective_function,
                                       values_over_iterations,
                                       sources, bounds, final_locations):
    """
    Visualize data and chosen locations

    objective_fun"""

    # Plot the objective values over time
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Objective function")
    plt.plot(values_over_iterations)
    if PAPER_READY:
        plt.savefig("vis/ObjectiveFunction.png")
    plt.title("Objective function values over time")
    plt.show()
    # compute all the function values

    visualize_sources(sources, final_locations)
    # visualize each time to alarm with the final locations
    [visualize_3D_with_highlights(XYZ=source.XYZ,
                                  metric=source.metric,
                                  highlight_locations=final_locations,
                                  is_parameterized=True,
                                  smoke_source=source)
                                  for source in sources]

    max_val = max([np.amax(source.metric) for source in sources])

    axis_labels = sources[0].axis_labels
    is_3d = sources[0].parameterized_locations.shape[1] == 3
    visualize_slices(objective_function, final_locations,
                     bounds, max_val=max_val, is_3d=is_3d,
                     axis_labels=axis_labels)


def run_multiobjective_problem(algorithm, function_type, verbose=False):
    """
    Actually perform the optimization

    problem

    function_type
    """
    # optimize the problem using 1,000 function evaluations
    # TODO should this be improved?
    optimization_logger.warning("Running for 1000 iterations. Could be tuned")
    algorithm.run(1000)

    if verbose:
        for solution in algorithm.result:
            print(
                "Solution : {}, Location : {}".format(
                    solution.objectives,
                    solution.variables))

    x_values = [s.objectives[1] for s in algorithm.result]
    plt.scatter(x_values,
                [s.objectives[0] for s in algorithm.result])

    if function_type == "multiobjective_competing":
        # invert the axis
        plt.xlim(max(x_values), min(x_values))
        plt.xlabel("time to false alarm")
    else:
        plt.xlabel("Number of detectors")

    plt.ylabel("The time to alarm")
    plt.title("Pareto optimality curve for the two functions")
    if PAPER_READY:
        optimization_logger.warning("Need to implment figure saving")
        plt.savefig(os.path.join("vis", f"{function_type}.png"))
    plt.show()


def make_bounds(bounds, sources, num_detectors):
    """Estimates bounds from data if needed, otherwise just reformats"""
    first_source = sources[0]
    min_values = np.amin(first_source.parameterized_locations, axis=0)
    max_values = np.amax(first_source.parameterized_locations, axis=0)
    bounds = [(min_values[i], max_values[i])
              for i in range(min_values.shape[0])]

    # duplicate the list that many times
    expanded_bounds = bounds * num_detectors

    return expanded_bounds
