import numpy as np
from scipy.interpolate import griddata
import pdb
import warnings
import logging

from platypus import (NSGAII, Problem, Real, Binary, CompoundOperator,
                      SBX, HUX, PM, BitFlip)

from .constants import (INTERPOLATION_METHOD, BIG_NUMBER,
                        SINGLE_OBJECTIVE_FUNCTIONS_MC,
                        SINGLE_OBJECTIVE_FUNCTIONS_TTA,
                        MULTI_OBJECTIVE_FUNCTIONS,
                        WORST_CASE_FUNCTIONS,
                        FASTEST_FUNCTIONS,
                        SECOND_FUNCTIONS)
from .constants import *


def make_location_objective(masked):
    """
    an example function to evalute the quality of the locations
    """
    if masked:
        def location_evaluation(xyons):  # TODO make this cleaner
            good = []
            for i in range(0, len(xyons), 3):
                x, y, on = xyons[i:i+3]
                if on[0]:
                    good.extend([x, y])
            if len(good) == 0:  # This means none were on
                return 0
            else:
                return np.linalg.norm(good)
    else:
        def location_evaluation(xys):
            return np.linalg.norm(xys)
    return location_evaluation


def make_counting_objective():
    """
    count the number of sources which are turned on
    """
    def counting_func(xyons):
        val = 0
        for i in range(0, len(xyons), 3):
            x, y, on = xyons[i:i+3]
            if on[0]:
                val += 1
        return val
    return counting_func


def make_lookup(simulated_points, metric, interpolation_method=INTERPOLATION_METHOD):
    """
    simulated_points : np.ArrayLike(Float), (n, m)
        n samples, m parameterizing dimensions
    metric : ArrayLike[Float]
        The metric corresponding to each of the locations. Likely represents time
        to alarm or max concentration
    interpolation_method : str
        The method for interpolating the data
    -----returns-----
    The sampled value, either using the nearest point or interpolation
    """
    num_dimensions = simulated_points.shape[1]  # The number of dimensions we are optimizing over

    def ret_func(query_point):  # this is what will be returned
        """
        query_point : ArrayLike[float]
            The point to get the value for
        """
        if len(query_point) != num_dimensions:
            raise ValueError("The number of dimensions of the query point was {} when it should have been {}".format(
                len(query_point), num_dimensions))

        interpolated_time = griddata(
            simulated_points, metric, query_point,
            method=interpolation_method)
        return interpolated_time
    return ret_func


def make_objective_function(
        sources,
        function_type="worst_case",
        interpolation_method="nearest",
        bad_sources=None,
        bounds=None):
    """A simple wrapper to just check what type to make"""

    # This should now work properly for single objective functions
    if function_type in SINGLE_OBJECTIVE_FUNCTIONS:
        objective_function = make_single_objective_function(
            sources,
            function_type=function_type,
            interpolation_method=interpolation_method)
    elif function_type in MULTI_OBJECTIVE_FUNCTIONS:
        objective_function = make_multiobjective_function(
            sources,
            bounds=bounds,
            function_type=function_type,
            interpolation_method=interpolation_method,
            bad_sources=bad_sources)
    else:
        raise ValueError(f"function type {function_type} not valid. Options are: {SINGLE_OBJECTIVE_FUNCTIONS + MULTI_OBJECTIVE_FUNCTIONS}")

    return objective_function


def make_single_objective_function(
        sources,
        verbose=False,
        function_type="worst_case_TTA",
        masked=False,
        interpolation_method="nearest"):
    """
    This function creates and returns the function which will be optimized
    -----inputs------
    sources : [SmokeSource]
        A list of smoke source objects representing all the sources
    verbose : Boolean
        print information during functinon evaluations
    type : String
        What function to use, "worst_cast", "softened", "second"
    masked : bool
        Is the input going to be [x, y, on, x, y, on, ...] representing active detectors
    interpolation_method : str
        Can be either nearest or linear, cubic is acceptable but idk why you'd do that
        How to interpolate the sampled points
    -----returns-----
    ret_func : Function[ArrayLike[Float] -> Float]
        This is the function which will eventually be optimized and it maps from the smoke detector locations to the time to alarm
        A function mapping [x1, y1, x2, y2, ....], represnting the x, y coordinates of each detector,to the objective function value
    """
    # Create data which will be used inside of the function to be returned
    funcs = []
    # The number of points parametrizing the space
    for source in sources:
        # create all of the functions mapping from a location to a time
        # This is notationionally dense but I think it is worthwhile
        # We are creating a list of functions for each of the smoke sources
        # The make_lookup function does that
        locations = source.parameterized_locations
        if function_type in SINGLE_OBJECTIVE_FUNCTIONS_TTA:
            metric = source.time_to_alarm
        elif function_type in SINGLE_OBJECTIVE_FUNCTIONS_MC:
            metric = source.max_concentration
        else:
            raise ValueError(f"function type {function_type}not supprted")

        funcs.append(make_lookup(locations, metric,
                                 interpolation_method=interpolation_method))
    dimensionality = locations.shape[1]

    def ret_func(locations):
        """
        xys : ArrayLike
            Could also be xyz
            Represents the x, y location of each of the smoke detectors as [x1, y1, x2, y2]
            could also be the [x, y, on, x, y, on,...] but masked should be specified in make_total_lookup_function
        -----returns-----
        worst_source : Float
            The objective function for the input
        """
        #print(f"locations : {locations}")
        # TODO this needs to be refactored
        if masked:
            vars_per_detector = dimensionality + 1
        else:
            vars_per_detector = dimensionality

        all_times = []  # each internal list coresponds to a smoke detector location

        for i in range(0, len(locations), vars_per_detector):
            detector_vars = locations[i:i+vars_per_detector]
            # The mask variable is the last one in the list for the detector
            # and it is itself a list of one boolean value, which must be
            # extracted
            if not masked or detector_vars[-1][0]:
                all_times.append([])
                for func in funcs:
                    if masked:
                        # don't include the mask variable
                        all_times[-1].append(func(detector_vars[:-1]))
                    else:
                        # include all variables
                        all_times[-1].append(func(detector_vars))

        if len(all_times) == 0:  # This means that no sources were turned on
            return BIG_NUMBER

        all_times = np.asarray(all_times)
        #print(f"all_times : {all_times}")
        #print(f"all_times.shape : {all_times.shape}")

        if function_type in WORST_CASE_FUNCTIONS:
            time_for_each_source = np.amin(all_times, axis=0)
            worst_source = np.amax(time_for_each_source)
            ret_val = worst_source
        elif function_type in SECOND_FUNCTIONS:
            time_for_each_source = np.amin(all_times, axis=0)
            second_source = np.sort(time_for_each_source)[1]
            ret_val = second_source
        elif function_type in FASTEST_FUNCTIONS:
            # print(all_times)
            # this just cares about the source-detector pair that alarms fastest
            ret_val = np.amin(all_times)
        else:
            raise ValueError(f"type is : {function_type} which is not included")
        if verbose:
            print(f"all of the times are {all_times}")
            print(
                f"The quickest detction for each source is {time_for_each_source}")
            print(f"The slowest-to-be-detected source takes {worst_source}")
        return ret_val
    return ret_func


def make_multiobjective_function(sources,
                                 bounds=None,
                                 function_type="worst_case",
                                 interpolation_method="nearest",
                                 bad_sources=None):
    """Make the multiobjective function"""

    if function_type == "multiobjective_counting":
        problem = make_multiobjective_function_counting(
            sources, bounds=bounds, interpolation_method=interpolation_method)
        algorithm = NSGAII(
            problem, variator=CompoundOperator(  # TODO look further into this
                SBX(), HUX(), PM(), BitFlip()))
    elif function_type == "multiobjective_competing":
        if bad_sources is None:
            raise ValueError(
                "specify bad_sources for multiobjective_competing")
        problem = make_multiobjective_function_competing(
            sources, bounds=bounds, bad_sources=bad_sources,
            interpolation_method=interpolation_method)  # TODO remove this
        algorithm = NSGAII(problem)
    else:
        raise ValueError(
            "The type : {} was not valid".format(function_type))

    return algorithm


def convert_to_spherical_from_points(X, Y, Z):
    # Make each of them lie on the range (-1, 1)
    X = np.expand_dims(normalize(X, -1, 2), axis=1)
    Y = np.expand_dims(normalize(Y, -1, 2), axis=1)
    Z = np.expand_dims(normalize(Z, -1, 2), axis=1)
    xyz = np.concatenate((X, Y, Z), axis=1)
    elev_az = xyz_to_spherical(xyz)[:, 1:]
    return elev_az[:, 0], elev_az[:, 1]


def normalize(x, lower_bound=0, scale=1):
    if scale <= 0:
        raise ValueError("scale was less than or equal to 0")
    minimum = np.min(x)
    maximum = np.max(x)
    diff = maximum - minimum
    x_prime = (x - minimum) / diff
    x_prime = x_prime * scale + lower_bound
    return x_prime


def xyz_to_spherical(xyz):
    """
    xyz : np.array
        this is (n, 3) with one row for each x, y, z
    modified from
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    r_elev_ax = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r_elev_ax[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    # for elevation angle defined from Z-axis down
    r_elev_ax[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    # ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    r_elev_ax[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return r_elev_ax


def spherical_to_xyz(elev_az):
    """
    elev_az : np.array
        This is a (n, 2) array where the columns represent the elevation and the azimuthal angles
    """
    # check that these aren't switched and migrate to all one convention
    phi = elev_az[:, 0]
    theta = elev_az[:, 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    xyz = np.vstack((x, y, z))
    return xyz.transpose()


def make_multiobjective_function_competing(
        sources, bounds=None, bad_sources=(), interpolation_method="nearest"):
    """Create an objective function with a false alarm"""

    # Create the two functions
    objective_function = make_single_objective_function(
        sources, interpolation_method=interpolation_method)  # the function to be optimized
    bad_objective_function = make_single_objective_function(
        bad_sources, function_type="worst_case_TTA",
        interpolation_method=interpolation_method)  # the function to be optimized

    def multiobjective_func(x):  # this is the double objective function
        return [objective_function(x), bad_objective_function(x)]

    parameterized_locations = sources[0].parameterized_locations
    dimensionality = parameterized_locations.shape[1]

    num_inputs = len(sources) * dimensionality
    NUM_OUPUTS = 2  # the default for now
    # define the demensionality of input and output spaces
    problem = Problem(num_inputs, NUM_OUPUTS)

    logging.warning(
        f"Creating a multiobjective competing function with dimensionality {dimensionality}")
    logging.warning(f"bounds are {bounds}")
    for i in range(dimensionality):
        # splat "*" notation is expanding the pair which is low, high
        problem.types[i::dimensionality] = Real(
            *bounds[i])  # This is the feasible region

    problem.function = multiobjective_func
    # the second function should be maximized rather than minimized
    problem.directions[1] = Problem.MAXIMIZE
    return problem


def make_multiobjective_function_counting(
        sources, bounds, times_more_detectors=1,
        interpolation_method="nearest"):
    """
    This balances the number of detectors with the quality of the outcome
    bounds : list[(x_min, x_max), (y_min, y_max), ...]
        The bounds on the feasible region
    """
    objective_function = make_single_objective_function(
        sources, interpolation_method=interpolation_method, masked=True)  # the function to be optimized
    counting_function = make_counting_objective()

    def multiobjective_func(x):  # this is the double objective function
        return [objective_function(x), counting_function(x)]
    # there is an x, y, and a mask for each source so there must be three
    # times more input variables
    # the upper bound on the number of detectors n times the number of
    # sources
    parameterized_locations = sources[0].parameterized_locations
    dimensionality = parameterized_locations.shape[1]

    # We add a boolean flag to each location variable
    num_inputs = len(sources) * (dimensionality + 1) * times_more_detectors
    NUM_OUPUTS = 2  # the default for now
    # define the demensionality of input and output spaces
    problem = Problem(num_inputs, NUM_OUPUTS)

    logging.warning(
        f"Creating a multiobjective counting function with dimensionality {dimensionality}")
    logging.warning(f"bounds are {bounds}")

    for i in range(dimensionality):
        # splat "*" notation is expanding the pair which is low, high
        problem.types[i::(dimensionality+1)] = Real(*bounds[i]
                                                    )  # This is the feasible region

    # indicator on whether the source is on
    problem.types[dimensionality::(dimensionality+1)] = Binary(1)
    problem.function = multiobjective_func
    return problem
