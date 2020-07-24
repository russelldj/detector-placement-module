# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = smokedetectoroptimization.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
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


def optimize(sources,
             num_detectors=1,
             function_type="worst_case",
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
        What function to optimize : 'worst_case', ''
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

    max_val = max([np.amax(source.time_to_alarm) for source in sources])

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

# def make_objective(smoke_sources, method=""):
#    """
#
#    Args:
#      smoke_sources List[Tuple[List[Float]]]: integer
#
#    Returns:
#      int: n-th Fibonacci number
#    """
#
#
# def get_time_to_alarm(data_file):
#    pass
#
#
# def parse_args(args):
#    """Parse command line parameters
#
#    Args:
#      args ([str]): command line parameters as list of strings
#
#    Returns:
#      :obj:`argparse.Namespace`: command line parameters namespace
#    """
#    parser = argparse.ArgumentParser(
#        description="Just a Fibonacci demonstration")
#    parser.add_argument(
#        "--version",
#        action="version",
#        version="SmokeDetectorOptimization {ver}".format(ver=__version__))
#    parser.add_argument(
#        dest="n",
#        help="n-th Fibonacci number",
#        type=int,
#        metavar="INT")
#    parser.add_argument(
#        "-v",
#        "--verbose",
#        dest="loglevel",
#        help="set loglevel to INFO",
#        action="store_const",
#        const=optimization_logger.INFO)
#    parser.add_argument(
#        "-vv",
#        "--very-verbose",
#        dest="loglevel",
#        help="set loglevel to DEBUG",
#        action="store_const",
#        const=optimization_logger.DEBUG)
#    return parser.parse_args(args)
#
#
# def setup_optimization_logger(loglevel):
#    """Setup basic optimization_logger
#
#    Args:
#      loglevel (int): minimum loglevel for emitting messages
#    """
#    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
#    optimization_logger.basicConfig(level=loglevel, stream=sys.stdout,
#                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")
#
#
# def main(args):
#    """Main entry point allowing external calls
#
#    Args:
#      args ([str]): command line parameter list
#    """
#    args = parse_args(args)
#    setup_optimization_logger(args.loglevel)
#    optimization_logger.debug("Starting crazy calculations...")
#    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
#    optimization_logger.info("Script ends here")
#
#
# def run():
#    """Entry point for console_scripts
#    """
#    main(sys.argv[1:])
#
#
# if __name__ == "__main__":
#    run()
