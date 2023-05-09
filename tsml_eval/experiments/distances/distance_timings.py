# -*- coding: utf-8 -*-
"""Classifier Experiments: code to run experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard format.
"""

__author__ = ["TonyBagnall"]

import os

import kmedoids

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
import time
from datetime import datetime

import numba
import numpy as np
if __name__ == "__main__":
    """Example simple usage, with args input via script or hard coded for testing."""
    numba.set_num_threads(1)

    tune = False
    normalise = True
    if (
            sys.argv is not None and sys.argv.__len__() > 1
    ):  # cluster run, this is fragile, requires all args atm
        result_dir = sys.argv[2]
    else:  # Local run
        print(" Local Run")  # noqa
        results_dir = "/home/chris/Documents/Results/temp/distances/"

    from aeon.distances import dtw_distance
    from aeon.distances.tests._utils import _time_distance, create_test_distance_numpy
    from aeon.distances import create_bounding_matrix
    from rust_dtw import dtw

    num_timepoints = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    num_ts_to_compute = 200

    dist = "dtw"
    for timepoints in num_timepoints:
        total = 0
        for i in range(num_ts_to_compute):
            x = np.random.rand(timepoints)
            y = np.random.rand(timepoints)
            start = time.time()
            rust_dtw = dtw(s=x, t=y, window=100, distance_mode="euclidean")
            # new = dtw_distance(x, y)
            total += time.time() - start
        print(f"{dist} {timepoints} {total}")
        joe = ""



    print("done")  # noqa

#  AEON
# dtw 1000 0.823296070098877
# dtw 2000 3.0499215126037598
# dtw 3000 7.33331298828125
# dtw 4000 12.681366205215454
# dtw 5000 20.102648973464966
# dtw 6000 29.033809423446655
# dtw 7000 39.83623456954956
# dtw 8000 52.48047089576721
# dtw 9000 66.02264523506165
# dtw 10000 81.39008808135986
# done