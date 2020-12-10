#! /usr/bin/env
from glob import glob
import pdb
import os

import pandas as pd
import numpy as np
import io
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
import pyvista as pv
import ubelt as ub

from .constants import (ALARM_THRESHOLD, PAPER_READY, NEVER_ALARMED_MULTIPLE, NUM_INTERPOLATION_SAMPLES)
from .functions import convert_to_spherical_from_points
from .visualization import visualize_metric, visualize_3D, visualize_3D_with_highlights

smoke_logger = logging.getLogger("smoke")


class SmokeSourceSet():
    """Represents a group of smoke sources for the same space"""

    def __init__(self,
                 data_paths,
                 mesh_file=None,
                 source_locations=None,
                 validate_geometry=False,
                 **kwargs):
        """
        data_paths : ArrayLike[str | path]
            The data to load
        mesh_file : str | path
            The mesh file
        source_locations: ArrayLike[ArrayLike]
            List of (x, y, z) lists representing the locations in 3D of the
            corresponding smoke source
        validate_geometry: bool
            Should you check that the XYZ is same
        **kwargs:
            Keyword arguments to be passed to the smoke source __init__ method
        """

        self.source_locations = source_locations

        if (self.source_locations is not None and
            len(self.source_locations) != len(data_paths)):
            raise ValueError("Length of data_paths and source_locations must be the same")

        if mesh_file is not None:
            self.mesh = pv.read(mesh_file)
        else:
            self.mesh = None

        self.sources = []
        for data_path in data_paths:
            # create a smoke source and then get it's time to alarm with a given parameterization
            print(f"Loading {data_path}")
            self.sources.append(SmokeSource(data_path,
                                       **kwargs))

        if validate_geometry:
            for i, source in enumerate(self.sources):
                for j in range(i):
                    other_source = self.sources[j].XYZ
                    if not np.allclose(source.XYZ.shape, other_source.shape):
                        raise ValueError(f"Data from {data_paths[i]} and {data_paths[j]} have different number of points")

                    if not np.allclose(source.XYZ, other_source):
                        raise ValueError(f"Data from {data_paths[i]} and {data_paths[j]} have different 3D geometry")



    def source_list(self):
        return self.sources

    def visualize_smoke_source(self, detector_locations=None, mesh_opacity=0.8,
                               plotter=pv.Plotter()):
        if self.mesh is None:
            raise ValueError("Can't visualize without a mesh file")

        plotter.add_mesh(self.mesh, opacity=mesh_opacity)

        if self.source_locations is not None:
            for source_location in self.source_locations:
                highlight = pv.Sphere(radius=0.15, center=source_location)
                plotter.add_mesh(highlight, color="red")

        if detector_locations is not None:
            for detector_location in detector_locations:
                highlight = pv.Sphere(radius=0.15, center=detector_location)
                plotter.add_mesh(highlight, color="green")

        plotter.show()



class SmokeSource():
    """Represents the smoke source and its metric values"""

    def __init__(self,
                 data_path,
                 mesh_file=None,
                 alarm_threshold=ALARM_THRESHOLD,
                 parameterization="phi_theta",
                 vis=False,
                 write_figs=PAPER_READY):
        """
        mesh_file: str | path
            Path to the mesh connecting the vertices
        """

        self.data_path = data_path

        self.concentrations = None
        self.alarmed = None
        self.XYZ = None
        self.unitized_XYZ = None
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

        if mesh_file is not None:
            self.mesh = pv.read(mesh_file)
        else:
            self.mesh = None

    def load_data(self, data_path):
        """Load the data from Fluent export. Checks file or dir"""

        smoke_logger.info(f"Beginning to load {data_path}")
        XYZ_cacher = ub.Cacher(f'{data_path}_XYZ',
            cfgstr=ub.hash_data('dependencies'))
        concentrations_cacher = ub.Cacher(f'{data_path}_concentration',
            cfgstr=ub.hash_data('dependencies'))

        self.XYZ = XYZ_cacher.tryload()
        self.concentrations = concentrations_cacher.tryload()

        if self.XYZ is None or self.concentrations is None:
            if os.path.isfile(data_path):
                self.load_file(data_path)
            elif os.path.isdir(data_path):
                self.load_directory(data_path)
            else:
                raise ValueError(
                    f"data path {data_path} was niether a directory nor a file.")
            assert self.concentrations is not None
            assert self.XYZ is not None
            XYZ_cacher.save(self.XYZ)
            concentrations_cacher.save(self.concentrations)

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

        self.concentrations = np.stack(concentrations_list).transpose()

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

        self.concentrations = np.stack(concentrations_list).transpose()

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
        self.max_concentration = np.amax(concentrations, axis=1)

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
        elif parameterization == "tsne":
            embedding = TSNE(n_components=2)
            self.parameterized_locations = embedding.fit_transform(self.XYZ)
            self.axis_labels = ("TSNE axis 1", "TSNE axis 2")
        elif parameterization == "mds":
            embedding = MDS(n_components=2)
            self.parameterized_locations = embedding.fit_transform(self.XYZ)
            self.axis_labels = ("MDS axis 1", "MDS axis 2")
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
            self.visualize_metric(write_figs=write_figs)

    def visualize_metric(self,
                         *,
                         interpolation=None,
                         num_samples=NUM_INTERPOLATION_SAMPLES,
                         write_figs=False,
                         points_fraction=1.0,
                         random_seed=0):
        """
        Visualize the metric value for this source
        interpolation : None | str
            What strategy to use for interpolation
        num_samples : int
            How many samples to use for interpolation in each dimension
        write_figs : bool
            Should you write the figure to vis/
        points_fraction : float
            between [0, 1], what randomly selected portion to visualize
        random_seed : int | None
            Random seed to ensure consistent results. Use None to get a random
            result
        """
        if points_fraction < 1.0:
            if points_fraction < 0:
                raise ValueError(f"Can't visualize a fraction {points_fraction} less than 0")

            num_points = self.metric.shape[0]
            np.random.seed(random_seed)
            sample_inds = np.random.choice(num_points,
                                           size=int(num_points * points_fraction))
            # Re-seed the generator so we don't interfere with future processes
            np.random.seed(None)
            parameterized_locations = self.parameterized_locations[sample_inds, :]
            metric = self.metric[sample_inds]
        else:
            parameterized_locations = self.parameterized_locations
            metric = self.metric

        visualize_metric(
            parameterized_locations,
            metric,
            interpolation=interpolation,
            num_samples=num_samples,
            axis_labels=self.axis_labels,
            write_figs=write_figs)

    def visualize_3D(self, *, which_metric=None, concentation_timestep=None,
                     log_concentrations=False, log_lower_bound=-12,
                     plotter=pv.Plotter, highlight_locations=None):
        """
        Visualize the smoke source in 3D

        metric : str
            {"time_to_alarm", "max_concentration"}, which metric to visualize
        concentation_timestep : int
            Which timestep to visualize the concentration for
        log_concentrations : bool
            Display the log concentrations
        log_lower_bound : float
            The lower bound for displaying logged values
        plotter : (pv.Plotter class) | (pv.PlotterITK class)
            This gives you an option to try interactive plotting with
            pv.PlotterITK. However, this severely limits other functionality
        highlight_locations : ArrayLike
            Locations in 3D as (x, y, z, x, y, z, ...)
        """
        if which_metric is None and concentation_timestep is None:
            smoke_logger.info("Showing the metric")
            metric = self.metric
            stitle = "Metric"
        elif which_metric == "time_to_alarm":
            smoke_logger.info("Showing the time to alarm")
            metric =  self.time_to_alarm
            stitle = "Time to alarm"
        elif which_metric == "max_concentration":
            smoke_logger.info("Showing the max concentration")
            metric = self.max_concentration
            stitle = "Max concentration"
        elif which_metric == "alarmed":
            smoke_logger.info("Showing alarmed")
            metric = self.alarmed[:, concentation_timestep]
            stitle = f"Alarmed at timestep {concentation_timestep}"
        elif which_metric == "index":
            smoke_logger.info("Showing alarmed")
            metric = np.arange(self.XYZ.shape[0])
            stitle = "Index"
        elif which_metric == "XYZ":
            smoke_logger.info("Showing alarmed")
            metric = self.get_unitized_XYZ()
            stitle = "Index"
        elif isinstance(concentation_timestep, int):
            timestep_concentrations = self.concentrations[:, concentation_timestep]

            if log_concentrations:
                timestep_concentrations = np.log10(timestep_concentrations)
                # Set negative invalid values to something low
                timestep_concentrations = np.nan_to_num(timestep_concentrations,
                                                        nan=log_lower_bound)
                timestep_concentrations = np.clip(timestep_concentrations,
                                                  a_min=log_lower_bound,
                                                  a_max=None)
            stitle = f"Log concentrations at timestep {concentation_timestep}" \
                     if log_concentrations else \
                     f"Concentrations at timestep {concentation_timestep}"
            metric = timestep_concentrations
        else:
            raise ValueError("Invalid arguments")

        instantiated_plotter = plotter()
        if self.mesh is not None:
            instantiated_plotter.add_mesh(self.mesh)
        return visualize_3D_with_highlights(self.XYZ,
                                           metric,
                                           stitle=stitle,
                                           highlight_locations=highlight_locations,
                                           plotter=instantiated_plotter,
                                           is_parameterized=False)

    def visualize_summary_statistics(self, quantiles=(0, 0.75, 0.9, 0.99, 0.999, 0.9999, 1)):
        """
        Show summary statistics about the smoke sources
        """
        data_quantiles = np.quantile(self.concentrations, quantiles, axis=0)
        for line, quant in zip(data_quantiles, quantiles):
            plt.plot(line, label=f"{quant}")
        plt.legend()
        plt.yscale("log")
        plt.xlabel("Timestep")
        plt.ylabel("Concentration")
        plt.title("Concentration ")
        plt.show()

        counts = np.count_nonzero(self.alarmed, axis=0)
        fraction = counts / self.alarmed.shape[0]
        plt.plot(fraction)
        plt.xlabel("Timestep")
        plt.ylabel("Fraction of locations exceeding threshold concentration")
        plt.show()

        self.time_to_alarm
        fraction_alarmed = []
        num_points, num_timesteps = self.concentrations.shape
        for timestep in range(num_timesteps):
            fraction_alarmed.append(np.count_nonzero(self.time_to_alarm <= timestep) / num_points)
        plt.plot(fraction_alarmed)
        plt.xlabel("Timestep")
        plt.ylabel("Fraction of points alarmed")
        plt.show()

    def visualize_parameterization(self, fraction=1):
        """
        Show the different parameterizations
        """
        self.visualize_3D(which_metric="XYZ")
        unitized_XYZ = self.get_unitized_XYZ()
        parameterized = self.parameterized_locations
        num_points = parameterized.shape[0]
        if fraction < 1:
            selected = np.random.choice(num_points,
                                        size=(int(num_points * fraction),))
            parameterized = parameterized[selected, :]
            unitized_XYZ = unitized_XYZ[selected, :]

        plt.scatter(parameterized[:, 0], parameterized[:, 1], c=unitized_XYZ)
        plt.xlabel(self.axis_labels[0])
        plt.ylabel(self.axis_labels[1])
        plt.title("Original unitized XYZ shown as RGB colors")
        plt.show()

    def compute_time_to_alarm(self, alarm_threshold):
        """This actually does the computation for time to alarm"""

        num_locations, num_timesteps = self.concentrations.shape
        smoke_logger.info(
            f'There are {num_timesteps} timesteps and' +
            f' {num_locations} locations')

        # Determine which entries have higher concentrations
        self.alarmed = self.concentrations > alarm_threshold
        nonzero = np.nonzero(self.alarmed)  # determine where the non zero entries
        # this is pairs indicating that it alarmed at that time and location
        nonzero_locations, nonzero_times = nonzero

        # TODO see if this can be vectorized
        time_to_alarm = []
        for loc in range(num_locations):  # All of the possible locations
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
        """
        Provide information about the closest XYZ points to a parameterized
        point
        """
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

    def get_closest_points(self, points, parameterized=True):
        """
        Return the XYZ point and corresponding parameterization for the nearest
        simulated point to the final optimized location.

        Does not currently accept masked data
        """
        if parameterized:
            dimensionality = self.get_parameterization_dimensionality()
        else:
            dimensionality = 3

        # Store the tuples of parameterized and XYZ locations
        closest_parameterized_XYZs = []
        for point in points:
            closest_parameterized_XYZs.append(
                self.get_closest_single_point(point,
                                              parameterized=parameterized))
        return closest_parameterized_XYZs

    def get_closest_single_point(self, point, parameterized=True):
        """
        Get the parameterized and coresponding XYZ point closest to the
        location
        """
        if parameterized:
            diffs = self.parameterized_locations - point
        else:
            diffs = self.XYZ - point

        dists = np.linalg.norm(diffs, axis=1)
        min_loc = np.argmin(dists)
        closest_parameterization = self.parameterized_locations[min_loc, :]
        closest_XYZ = self.XYZ[min_loc, :]

        return {"parameterized" : closest_parameterization,
                "XYZ" : closest_XYZ,
                "index" : min_loc}

    def get_parameterization_dimensionality(self):
        """get the parameterization of the underlying parameterization"""
        if self.parameterized_locations is None:
            raise ValueError("Load data first")

        return self.parameterized_locations.shape[1]

    def get_unitized_XYZ(self):
        if self.unitized_XYZ is not None:
            return self.unitized_XYZ

        if self.XYZ is None:
            raise ValueError("Load data first")

        mins = self.XYZ.min(axis=0)
        ptp = self.XYZ.ptp(axis=0)
        self.unitized_XYZ = (self.XYZ - mins) / ptp
        return self.unitized_XYZ

    def set_infeasible(self, infeasible_region):
        """Set some region of the XYZ space as infeasible"""
        raise NotImplementedError()

    def augment(self, augmentation):
        """Change the data in some way for testing purposes"""
        raise NotImplementedError()
