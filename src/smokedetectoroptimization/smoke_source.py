#! /usr/bin/env
import pandas as pd
import os
import numpy as np
from glob import glob
import pdb
import io
import logging

from .constants import (ALARM_THRESHOLD, PAPER_READY, NEVER_ALARMED_MULTIPLE)
from .functions import convert_to_spherical_from_points
from .visualization import visualize_time_to_alarm

smoke_logger = logging.getLogger("smoke")


class SmokeSource():
    """Represents the smoke source and its time to alarm"""

    def __init__(self,
                 data_path,
                 alarm_threshold=ALARM_THRESHOLD,
                 parameterization="phi_theta",
                 vis=False,
                 write_figs=PAPER_READY):
        self.data_path = data_path

        self.concentrations = None
        self.XYZ = None
        self.parameterized_locations = None
        self.axis_labels = None
        self.time_to_alarm = None
        self.max_concentration = None
        self.metric = None

        self.load_data(data_path)
        self.get_time_to_alarm(
            alarm_threshold=alarm_threshold,
            parameterization=parameterization,
            vis=vis,
            write_figs=write_figs)

    def load_data(self, data_path):
        """Load the data from Fluent export. Checks file or dir"""
        smoke_logger.info(f"Beginning to load {data_path}")
        if os.path.isfile(data_path):
            self.load_file(data_path)
        elif os.path.isdir(data_path):
            self.load_directory(data_path)
        else:
            raise ValueError(
                f"data path {data_path} was niether a directory nor a file.")
        assert self.concentrations is not None
        assert self.XYZ is not None
        smoke_logger.info(f"done loading {data_path}")

    def load_file(self, data_file):
        """
        data_file : string
            This is the path to the data file as exported by the
            Fluent simulation
        """
        # This is a doubly-nested list with each internal list representing a
        # single timestep
        all_points = [[]]
        # read through line by line and parse by the time
        with open(data_file, 'r') as infile:
            for line in infile:
                if "symmetry" in line:
                    all_points.append([line])
                all_points[-1].append(line)

        concentrations_list = []

        for time in all_points[1:-1]:
            timestep_data = pd.read_csv(io.StringIO('\n'.join(time[4:])))
            # drop the last row which is always none
            timestep_data.drop(timestep_data.tail(1).index, inplace=True)
            # get all of the concentrations but the null last one
            concentrations_list.append(
                timestep_data['dpm-concentration'].values)

        # spatial locations are assumed to be constant over time
        # so only the last ones are taken
        self.XYZ = np.stack((timestep_data['x-coordinate'].values,
                             timestep_data['y-coordinate'].values,
                             timestep_data['z-coordinate'].values), axis=1)

        self.concentrations = np.stack(concentrations_list)

    def load_directory(self, data_directory):
        """
        directory : String
            The folder containing the data files
            Each of the files should represent a timestep and they must be
            alphabetized
        -----returns-----
        data : TODO
        """
        files_pattern = os.path.join(data_directory, "*")
        filenames = sorted(glob(files_pattern))
        if len(filenames) == 0:
            raise ValueError(f"There were no files in : {data_directory}")
        concentrations_list = []
        # read each of the filenames, each one representing a timestep
        for filename in filenames:
            # read a space-delimited file
            timestep_data = pd.read_csv(
                filename, sep=' ', skipinitialspace=True)
            # The last column is junk
            timestep_data.drop(
                labels=timestep_data.columns[-1], inplace=True, axis=1)
            concentrations_list.append(
                timestep_data['dpm-concentration'].values)

        # spatial locations are assumed to be constant over time
        # so only the last ones are taken
        self.XYZ = np.stack((timestep_data['x-coordinate'].values,
                             timestep_data['y-coordinate'].values,
                             timestep_data['z-coordinate'].values), axis=1)

        self.concentrations = np.stack(concentrations_list)

    # Might be the first victim for @extern
    def get_time_to_alarm(
            self,
            alarm_threshold=ALARM_THRESHOLD,
            parameterization="phi_theta",
            metric="time_to_alarm",
            vis=False,
            write_figs=PAPER_READY):
        """
        Gets time to alarm and performs augmentations

        alarm_threshold : Float
            What concentraion will trigger the detector
        parameterization : str
            'x_y' 'y_z' 'x_z' 'x_y_z' 'phi_theta' How to parameterize the data
        metric : str
            "time_to_alarm", "max_consentration".
        vis : Boolean
            Should it be shown
        write_figs : Boolean
            Should you write out figures to ./vis/

        Returns None
        """

        self.time_to_alarm, concentrations = self.compute_time_to_alarm(
            alarm_threshold)
        _, num_samples = concentrations.shape
        self.max_concentration = np.amax(concentrations, axis=0)

        if parameterization == "xy":
            self.parameterized_locations = self.XYZ[:, :2].copy()
            self.axis_labels = ("x locations", "y locations")
        elif parameterization == "yz":
            self.parameterized_locations = self.XYZ[:, 1:].copy()
            self.axis_labels = ("y locations", "z locations")
        elif parameterization == "xz":
            # take the 0th and 2nd columns
            self.parameterized_locations = self.XYZ[:, 0:4:2].copy()
            self.axis_labels = ("x locations", "z locations")
        elif parameterization == "xyz":
            self.parameterized_locations = self.XYZ.copy()
            self.axis_labels = ("x locations", "y locations", "z locations")
        elif parameterization == "phi_theta":
            phi, theta = convert_to_spherical_from_points(
                self.XYZ[:, 0], self.XYZ[:, 1], self.XYZ[:, 2])
            self.parameterized_locations = np.stack((phi, theta), axis=1)
            self.axis_labels = ("phi locations", "theta locations")
        else:
            raise ValueError(
                f"parameterization {parameterization} wasn't valid")

        if metric == "time_to_alarm":
            self.metric = self.time_to_alarm
        elif metric == "max_concentration":
            self.metric = self.max_concentration
        else:
            raise ValueError(f"metric {metric} not suppored")

        if vis:
            visualize_time_to_alarm(
                self.parameterized_locations,
                self.time_to_alarm,
                num_samples=num_samples,
                concentrations=concentrations,
                axis_labels=self.axis_labels,
                write_figs=write_figs)

    def compute_time_to_alarm(self, alarm_threshold):
        """This actually does the computation for time to alarm"""

        # Get all of the concentrations
        num_timesteps = self.concentrations.shape[0]
        # TODO make this a logger again
        smoke_logger.info(
            f'There are {self.concentrations.shape[0]} timesteps and' +
            f' {self.concentrations.shape[1]} locations')

        # Determine which entries have higher concentrations
        num_timesteps = self.concentrations.shape[0]
        alarmed = self.concentrations > alarm_threshold
        nonzero = np.nonzero(alarmed)  # determine where the non zero entries
        # this is pairs indicating that it alarmed at that time and location
        nonzero_times, nonzero_locations = nonzero

        # TODO see if this can be vectorized
        time_to_alarm = []
        for loc in range(alarmed.shape[1]):  # All of the possible locations
            # the indices for times which have alarmed at that location
            same = (loc == nonzero_locations)
            if np.any(same):  # check if this alarmed at any pointh
                # These are all of the times which alarmed
                alarmed_times = nonzero_times[same]
                # Determine the first alarming time
                time_to_alarm.append(min(alarmed_times))
            else:
                # this represents a location which was never alarmed
                time_to_alarm.append(num_timesteps * NEVER_ALARMED_MULTIPLE)

        # Perform the augmentations
        time_to_alarm = np.array(time_to_alarm)
        return time_to_alarm, self.concentrations

    def describe_closest_points(self, parameterized_points):
        closest_parameterized_XYZs = self.get_closest_points(
            parameterized_points)
        for closest_point in closest_parameterized_XYZs:
            parameterized_point, XYZ_point = closest_point
            tags = []
            for i, axis_label in enumerate(self.axis_labels):
                tags.append(f"{axis_label[:-1]} : {parameterized_point[i]}")
            parmaterized_description = ", ".join(tags)
            print(f"Parameterized, {parmaterized_description}")

            tags = []
            for i, axis in enumerate(["X", "Y", "Z"]):
                tags.append(f"{axis} : {XYZ_point[i]}")
            threeD_description = ", ".join(tags)
            print(f"3D, {threeD_description}")
            print("------------")

    def get_closest_points(self, points, parametrized =True):
        """
        Return the XYZ point and corresponding parameterization for the nearest
        simulated point to the final optimized location.

        Does not currently accept masked data
        """
        if parametrized:
            dimensionality = self.get_parameterization_dimensionality()
        else:
            dimensionality = 3

        # Store the tuples of parameterized and XYZ locations
        closest_parameterized_XYZs = []
        for i in range(0, len(points), dimensionality):
            point = points[i:i + dimensionality]
            closest_parameterized_XYZs.append(
                self.get_closest_single_point(point,
                                              parametrized=parametrized))
        return closest_parameterized_XYZs
        # closest_parameterized_points, closest_XYZs = list(
        #    zip(parameterized_XYZs))
        # return closest_parameterized_points, closest_XYZs

    def get_closest_single_point(self, point, parametrized=True):
        """
        Get the parameterized and coresponding XYZ point closest to the
        location
        """
        if parametrized:
            diffs = self.parameterized_locations - point
        else:
            diffs = self.XYZ - point

        dists = np.linalg.norm(diffs, axis=1)
        min_loc = np.argmin(dists)
        closest_parameterization = self.parameterized_locations[min_loc, :]
        closest_XYZ = self.XYZ[min_loc, :]

        return {"parameterized" : closest_parameterization,
                "XYZ" : closest_XYZ}

    def get_parameterization_dimensionality(self):
        """get the parameterization of the underlying parameterization"""
        if self.parameterized_locations is None:
            raise ValueError("Load data first")

        return self.parameterized_locations.shape[1]

    def set_infeasible(self, infeasible_region):
        """Set some region of the XYZ space as infeasible"""
        raise NotImplementedError()

    def augment(self, augmentation):
        """Change the data in some way for testing purposes"""
        raise NotImplementedError()
