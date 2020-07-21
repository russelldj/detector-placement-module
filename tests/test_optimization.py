# -*- coding: utf-8 -*-

import pytest
import pdb

from smokedetectoroptimization.optimizer import optimize
from smokedetectoroptimization.smoke_source import SmokeSource

__author__ = "David Russell"
__copyright__ = "David Russell"
__license__ = "mit"


def test_optimization():
    """test data loading"""
    DATA_DIRS = ["../data/input/first_computer_full3D",
                 "../data/input/second_computer_full3D"]
    sources = [SmokeSource(data_dir).get_time_to_alarm(method="phi_theta")
               for data_dir in DATA_DIRS]
    optimize(sources, num_detectors=1, function_type="worst_case")
    optimize(sources, num_detectors=1, function_type="second")
    optimize(sources, num_detectors=1, function_type="fastest")
    optimize(sources, num_detectors=1, function_type="multiobjective_counting")


test_optimization()
