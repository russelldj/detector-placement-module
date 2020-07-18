# -*- coding: utf-8 -*-

import pytest
from smokedetectoroptimization.smoke_source import SmokeSource

__author__ = "David Russell"
__copyright__ = "David Russell"
__license__ = "mit"


def test_smoke_source():
    """test data loading"""
    DATA_DIRS = ["../data/input/first_computer_full3D"]
    for data_dir in DATA_DIRS:
        s = SmokeSource(data_dir)
        print(s.concentrations)
        xs, ys, zs, time_to_alarm = s.get_time_to_alarm(method="xy")
        print(xs, ys, zs, time_to_alarm)
        xs, ys, zs, time_to_alarm = s.get_time_to_alarm(method="yz")
        xs, ys, zs, time_to_alarm = s.get_time_to_alarm(method="xz")
        xs, ys, zs, time_to_alarm = s.get_time_to_alarm(method="xyz")
        xs, ys, zs, time_to_alarm = s.get_time_to_alarm(method="phi_theta")


test_smoke_source()
