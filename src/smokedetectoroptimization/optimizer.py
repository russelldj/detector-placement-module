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

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

from .functions import make_objective_function
from .constants import (SINGLE_OBJECTIVE_FUNCTIONS, MULTI_OBJECTIVE_FUNCTIONS,
                        PAPER_READY)
from .visualization import visualize_slices, visualize_sources

from smokedetectoroptimization import __version__


__author__ = "David Russell"
__copyright__ = "David Russell"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def optimize(sources,
             num_detectors,
             function_type="worst_case",
             bounds=None,
             bad_sources=None,
             vis=True,
             interpolation_method="nearest"):
    """
    sources : ArrayLike
        list of (x, y, time) tuples
    num_detectors : int
        The number of detectors to place
    bounds : ArrayLike
        [x_low, x_high, y_low, y_high] or [x_low, x_high, y_low, y_high, z_low, z_high]
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

    bounds = make_bounds(bounds, sources, num_detectors)

    # compute the bounds
    objective_function = make_objective_function(
        sources=sources,
        bounds=bounds,
        function_type=function_type,
        bad_sources=bad_sources,
        interpolation_method=interpolation_method)  # This is keyword arguments which are passed from the call to optimize

    if function_type in SINGLE_OBJECTIVE_FUNCTIONS:
        # Do the single objective optimization
        res, values_over_iterations = run_single_objective_problem(
            objective_function, bounds, compute_function_values=vis)

        if vis:
            visualize_single_objective_problem(objective_function,
                                               values_over_iterations,
                                               sources, bounds, res.x)

    elif function_type in MULTI_OBJECTIVE_FUNCTIONS:
        logging.warning("Not computing multiobjective function")


def run_single_objective_problem(objective_function, bounds,
                                 compute_function_values=False):
    """
    Perform single objective optimization

    objective_function : func(ArrayLike) -> float
    bounds : bounds
    compute_function_values, bool : compute the values of the function at each iteration
    """

    locations_over_iterations = []

    def callback(xk, convergence):
        locations_over_iterations.append(xk)

    # This is where the actual optimization occurs
    res = differential_evolution(objective_function, bounds, callback=callback)

    if compute_function_values:
        logging.warning(
            f"Starting to compute values for plotting. Might be expensive.")
        values_over_iterations = [objective_function(
            x) for x in locations_over_iterations]
        logging.warning(f"Done computing values for plotting")

        return res, values_over_iterations
    else:
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

    max_val = visualize_sources(sources, final_locations)

    is_3d = sources[0][2] is not None
    visualize_slices(objective_function, final_locations,
                     bounds, max_val=max_val, is_3d=is_3d)


def run_optimizer(objective_function, optimizer_type=""):
    """
    Actually perform the optimization

    optimizer
    """

    if multiobjective:
        if multiobjective_type == "counting":
            problem = self.make_platypus_objective_function_counting(
                sources)  # TODO remove this
            # it complains about needing a defined mutator for mixed problems
            # Suggestion taken from
            # https://github.com/Project-Platypus/Platypus/issues/31
            algorithm = NSGAII(
                problem, variator=CompoundOperator(
                    SBX(), HUX(), PM(), BitFlip()))
            second_objective = "The number of detectors"
            savefile = "vis/ParetoNumDetectors.png"
        elif multiobjective_type == "competing_function":
            if "bad_sources" not in kwargs:
                raise ValueError(
                    "bad_sources should have been included in the kwargs")
            bad_sources = kwargs["bad_sources"]
            problem = self.make_platypus_objective_function(
                sources, "competing_function", bad_sources=bad_sources)  # TODO remove this
            algorithm = NSGAII(problem)
            second_objective = "The time to alarm for the exercise equiptment"
            savefile = "vis/ParetoExerciseFalseAlarm.png"
        else:
            raise ValueError(
                "The type : {} was not valid".format(multiobjective_type))
        # optimize the problem using 1,000 function evaluations
        # TODO should this be improved?
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
        plt.xlabel(second_objective)
        if second_objective == "competing_function":
            # invert the axis
            plt.set_xlim(max(x_values), min(x_values))

        plt.ylabel("The time to alarm")
        plt.title("Pareto optimality curve for the two functions")
        if PAPER_READY:
            plt.savefig(savefile)
        plt.show()
        res = algorithm
        if visualize:
            logging.warn(
                "Can't visualize the objective values for a multiobjective run",
                UserWarning)


def make_bounds(bounds, sources, num_detectors):
    """Estimates bounds from data if needed, otherwise just reformats"""
    # TODO this should be np.allclose when looping over all sources
    # Should support the 3D case
    if bounds is None:
        X = sources[0][0]
        Y = sources[0][1]
        Z = sources[0][2]
        bounds = [(np.min(X), np.max(X)), (np.min(Y), np.max(Y))]
        if Z is not None:
            bounds.append(([np.min(Z), np.max(Z)]))

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
#        const=logging.INFO)
#    parser.add_argument(
#        "-vv",
#        "--very-verbose",
#        dest="loglevel",
#        help="set loglevel to DEBUG",
#        action="store_const",
#        const=logging.DEBUG)
#    return parser.parse_args(args)
#
#
# def setup_logging(loglevel):
#    """Setup basic logging
#
#    Args:
#      loglevel (int): minimum loglevel for emitting messages
#    """
#    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
#    logging.basicConfig(level=loglevel, stream=sys.stdout,
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
#    setup_logging(args.loglevel)
#    _logger.debug("Starting crazy calculations...")
#    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
#    _logger.info("Script ends here")
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
