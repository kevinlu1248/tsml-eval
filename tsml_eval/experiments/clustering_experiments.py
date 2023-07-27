"""Clustering Experiments: code for experiments as an alternative to orchestration.

This file is configured for runs of the main method with command line arguments, or for
single debugging runs. Results are written in a standard tsml format.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

import os
import warnings

import numpy as np

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys

import numba

from tsml_eval.experiments import run_clustering_experiment
from tsml_eval.experiments.experiments import _check_existing_results, _load_data
from tsml_eval.experiments.set_clusterer import set_clusterer
from tsml_eval.utils.experiments import (
    _results_present,
    assign_gpu,
    stratified_resample_data,
)


def run_experiment(args, overwrite=False):
    """Mechanism for testing clusterers on the UCR data format.

    This mirrors the mechanism used in the Java based tsml. Results generated using the
    method are in the same format as tsml and can be directly compared to the results
    generated in Java.
    """
    numba.set_num_threads(1)

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        try:
            gpu = assign_gpu()
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"Assigned GPU {gpu} to process.")
        except Exception:
            print("Unable to assign GPU to process.")

    # cluster run (with args), this is fragile
    if args is not None and args.__len__() > 1:
        print("Input args = ", args)
        data_path = args[1]
        results_path = args[2]
        clusterer_name = args[3]
        dataset = args[4]
        resample_id = int(args[5])

        if len(args) > 6:
            test_fold = args[6].lower() == "false"
        else:
            test_fold = True

        if len(args) > 7:
            predefined_resample = args[7].lower() == "true"
        else:
            predefined_resample = False

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and _results_present(
            results_path,
            clusterer_name,
            dataset,
            resample_id=resample_id,
            split="BOTH" if test_fold else "TRAIN",
        ):
            print("Ignoring, results already present")
        else:
            build_test_file, build_train_file = _check_existing_results(
                results_path,
                clusterer_name,
                dataset,
                resample_id,
                overwrite,
                True,
                True,
            )
            if not build_test_file and not build_train_file:
                warnings.warn(
                    "All files exist and not overwriting, skipping.", stacklevel=1
                )
                return
            X_train, y_train, X_test, y_test, resample_id = _load_data(
                data_path, dataset, resample_id, predefined_resample
            )
            # Normalise, temporary fix: Make this a a parameter and
            # use an aeon transformer
            X_train = X_train.squeeze()
            X_test = X_test.squeeze()
            normalise = True
            if normalise:
                from sklearn.preprocessing import StandardScaler

                s = StandardScaler()
                X_train = s.fit_transform(X_train.T)
                X_train = X_train.T
                X_test = s.fit_transform(X_test.T)
                X_test = X_test.T
            if resample_id > 0:
                X_train, y_train, X_test, y_test = stratified_resample_data(
                    X_train, y_train, X_test, y_test, random_state=resample_id
                )
            # Get number of clusters
            n_clusters = len(np.unique(y_test))
            # Set up kwarg parameters: these are the distance defaults
            paras = {"n_clusters": n_clusters}
            # Pass to set_clusterer
            clusterer = set_clusterer(clusterer_name, **paras)
            run_clustering_experiment(
                X_train,
                y_train,
                clusterer,
                results_path,
                X_test=X_test,
                y_test=y_test,
                clusterer_name=clusterer_name,
                dataset_name=dataset,
                resample_id=resample_id,
                build_train_file=build_train_file,
                build_test_file=build_test_file,
            )
    # local run (no args)
    else:
        # These are example parameters, change as required for local runs
        # Do not include paths to your local directories here in PRs
        # If threading is required, see the threaded version of this file
        data_path = "c://data//"
        results_path = "C://temp//"
        clusterer_name = "kmeans-dtw"
        dataset = "ItalyPowerDemand"
        resample_id = 0
        predefined_resample = False
        n_jobs = 4
        overwrite = False
        build_test_file = True
        clusterer = set_clusterer(
            clusterer_name, random_state=resample_id, n_jobs=n_jobs
        )
        print(f"Local Run of {clusterer_name} ({clusterer.__class__.__name__}).")

        build_test_file, build_train_file = _check_existing_results(
            results_path,
            clusterer_name,
            dataset,
            resample_id,
            overwrite,
            build_test_file,
            True,
        )
        if not build_test_file and not build_train_file:
            warnings.warn(
                "All files exist and not overwriting, skipping.", stacklevel=1
            )
            return
        X_train, y_train, X_test, y_test, temp = _load_data(
            data_path, dataset, resample_id, predefined_resample
        )
        # Normalise, could make a parameter
        # Normalise, temporary fix: Make this a a parameter and
        # use an aeon transformer
        X_train = X_train.squeeze()
        X_test = X_test.squeeze()
        normalise = True
        if normalise:
            from sklearn.preprocessing import StandardScaler

            s = StandardScaler()
            X_train = s.fit_transform(X_train.T)
            X_train = X_train.T
            X_test = s.fit_transform(X_test.T)
            X_test = X_test.T

        if resample_id > 0:
            X_train, y_train, X_test, y_test = stratified_resample_data(
                X_train, y_train, X_test, y_test, random_state=resample_id
            )
            # Get number of clusters
        n_clusters = len(np.unique(y_test))
        # Set up kwarg parameters: these are the distance defaults
        paras = {"n_clusters": n_clusters}
        # Pass to set_clusterer
        clusterer = set_clusterer(clusterer_name, **paras)
        run_clustering_experiment(
            X_train,
            y_train,
            clusterer,
            results_path,
            X_test=X_test,
            y_test=y_test,
            clusterer_name=clusterer_name,
            dataset_name=dataset,
            resample_id=resample_id,
            build_train_file=build_train_file,
            build_test_file=build_test_file,
        )


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
