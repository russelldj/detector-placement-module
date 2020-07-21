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


class SmokeSource():
    """Represents the smoke source and its time to alarm"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data(data_path)

    def load_data(self, data_path):
        """Load the data from Fluent export. Checks file or dir"""
        logging.warning(f"beginging to load {data_path}")
        if os.path.isfile(data_path):
            self.load_file(data_path)
        elif os.path.isdir(data_path):
            self.load_directory(data_path)
        else:
            raise ValueError(
                f"data path {data_path} was niether a directory nor a file.")
        assert self.concentrations is not None
        logging.warning(f"done loading {data_path}")

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
        self.xs = timestep_data['x-coordinate'].values
        self.ys = timestep_data['y-coordinate'].values
        self.zs = timestep_data['z-coordinate'].values

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
        self.xs = timestep_data['x-coordinate'].values
        self.ys = timestep_data['y-coordinate'].values
        self.zs = timestep_data['z-coordinate'].values

        self.concentrations = np.stack(concentrations_list)

    # Might be the first victim for @extern
    def get_time_to_alarm(
            self,
            alarm_threshold=ALARM_THRESHOLD,
            visualize=False,
            method="phi_theta",
            write_figs=PAPER_READY):
        """
        Gets time to alarm and performs augmentations

        alarm_threshold : Float
            What concentraion will trigger the detector
        visualize : Boolean
            Should it be shown
        spherical_projection : Boolean
            Should the data be projected into spherical coordinates
        write_figs : Boolean
            Should you write out figures to ./vis/
        method : str
            'xy' 'yz' 'xz' 'xyz' 'phi_theta' How to parameterize the data

        Returns (xs, ys, zs, time_to_alarms)
            z coorinates may be None
        """

        time_to_alarm, concentrations = self.compute_time_to_alarm(
            alarm_threshold)
        num_timesteps, num_samples = concentrations.shape

        if method == "xy":
            xs = self.xs.copy()
            ys = self.ys.copy()
            zs = None
            axis_labels = ("x locations", "y locations")
        elif method == "yz":
            xs = self.ys.copy()
            ys = self.zs.copy()
            zs = None
            axis_labels = ("y locations", "z locations")
        elif method == "xz":
            xs = self.xs.copy()
            ys = self.zs.copy()
            zs = None
            axis_labels = ("x locations", "z locations")
        elif method == "xyz":
            xs = self.xs.copy()
            ys = self.ys.copy()
            zs = self.zs.copy()
            axis_labels = ("x locations", "y locations", "z locations")
        elif method == "phi_theta":
            xs, ys = convert_to_spherical_from_points(self.xs, self.ys,
                                                      self.zs)
            zs = None
            axis_labels = ("phi locations", "theta locations")
        else:
            raise ValueError(f"method {method} wasn't valid")

        if visualize:
            visualize_time_to_alarm(
                xs, ys, zs, time_to_alarm, num_samples=num_samples,
                concentrations=concentrations, axis_labels=axis_labels,
                write_figs=write_figs)

        return (xs, ys, zs, time_to_alarm)

    def compute_time_to_alarm(self, alarm_threshold):
        """This actually does the computation for time to alarm"""

        # Get all of the concentrations
        num_timesteps = self.concentrations.shape[0]
        # TODO make this a logger again
        logging.warning(
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

    def set_infeasible(self, infeasible_region):
        """Set some region of the XYZ space as infeasible"""
        raise NotImplementedError()

    def augment(self, augmentation):
        """Change the data in some way for testing purposes"""
        raise NotImplementedError()
