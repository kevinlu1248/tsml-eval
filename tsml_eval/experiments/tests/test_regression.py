"""Tests for regression experiments."""

__author__ = ["MatthewMiddlehurst"]

import os
import runpy

import pytest
from tsml.dummy import DummyClassifier

from tsml_eval.experiments import (
    regression_experiments,
    run_regression_experiment,
    set_regressor,
    threaded_regression_experiments,
)
from tsml_eval.experiments.tests import _REGRESSOR_RESULTS_PATH
from tsml_eval.utils.test_utils import (
    _TEST_DATA_PATH,
    _check_set_method,
    _check_set_method_results,
)
from tsml_eval.utils.tests.test_results_writing import _check_regression_file_format


@pytest.mark.parametrize(
    "regressor",
    ["DummyRegressor-tsml", "DummyRegressor-aeon", "DummyRegressor-sklearn"],
)
@pytest.mark.parametrize(
    "dataset",
    ["MinimalGasPrices", "UnequalMinimalGasPrices", "MinimalCardanoSentiment"],
)
def test_run_regression_experiment(regressor, dataset):
    """Test regression experiments with test data and regressor."""
    if regressor == "DummyRegressor-aeon" and dataset == "UnequalMinimalGasPrices":
        return  # todo remove when aeon dummy supports unequal

    args = [
        _TEST_DATA_PATH,
        _REGRESSOR_RESULTS_PATH,
        regressor,
        dataset,
        "0",
        "-tr",
    ]

    regression_experiments.run_experiment(args)

    test_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample0.csv"
    )
    train_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/trainResample0.csv"
    )

    assert os.path.exists(test_file) and os.path.exists(train_file)

    _check_regression_file_format(test_file)
    _check_regression_file_format(train_file)

    # test present results checking
    regression_experiments.run_experiment(args)

    os.remove(test_file)
    os.remove(train_file)


def test_run_regression_experiment_main():
    """Test regression experiments main with test data and regressor."""
    regressor = "ROCKET"
    dataset = "MinimalGasPrices"

    # run twice to test results present check
    for _ in range(2):
        runpy.run_path(
            "./tsml_eval/experiments/regression_experiments.py"
            if os.getcwd().split("\\")[-1] != "tests"
            else "../regression_experiments.py",
            run_name="__main__",
        )

    test_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample0.csv"
    )
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    os.remove(
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample0.csv"
    )


def test_run_threaded_regression_experiment():
    """Test threaded regression experiments with test data and regressor."""
    regressor = "ROCKET"
    dataset = "MinimalGasPrices"

    args = [
        _TEST_DATA_PATH,
        _REGRESSOR_RESULTS_PATH,
        regressor,
        dataset,
        "1",
        "-nj",
        "2",
        # also test normalisation here
        "--row_normalise",
    ]

    threaded_regression_experiments.run_experiment(args)

    test_file = (
        f"{_REGRESSOR_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample1.csv"
    )
    assert os.path.exists(test_file)
    _check_regression_file_format(test_file)

    # test present results checking
    threaded_regression_experiments.run_experiment(args)

    # this covers the main method and experiment function result file checking
    runpy.run_path(
        "./tsml_eval/experiments/threaded_regression_experiments.py"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../threaded_regression_experiments.py",
        run_name="__main__",
    )

    os.remove(test_file)


def test_run_regression_experiment_invalid_build_settings():
    """Test run_regression_experiment method with invalid build settings."""
    with pytest.raises(ValueError, match="Both test_file and train_file"):
        run_regression_experiment(
            [],
            [],
            [],
            [],
            None,
            "",
            build_test_file=False,
            build_train_file=False,
        )


def test_run_regression_experiment_invalid_estimator():
    """Test run_regression_experiment method with invalid estimator."""
    with pytest.raises(TypeError, match="regressor must be a"):
        run_regression_experiment(
            [],
            [],
            [],
            [],
            DummyClassifier(),
            "",
        )


def test_set_regressor():
    """Test set_regressor method."""
    regressor_lists = [
        set_regressor.convolution_based_regressors,
        set_regressor.deep_learning_regressors,
        set_regressor.dictionary_based_regressors,
        set_regressor.distance_based_regressors,
        set_regressor.feature_based_regressors,
        set_regressor.hybrid_regressors,
        set_regressor.interval_based_regressors,
        set_regressor.other_regressors,
        set_regressor.shapelet_based_regressors,
        set_regressor.vector_regressors,
    ]

    regressor_dict = {}
    all_regressor_names = []

    for regressor_list in regressor_lists:
        _check_set_method(
            set_regressor.set_regressor,
            regressor_list,
            regressor_dict,
            all_regressor_names,
        )

    _check_set_method_results(
        regressor_dict, estimator_name="Regressors", method_name="set_regressor"
    )


def test_set_regressor_invalid():
    """Test set_regressor method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN REGRESSOR"):
        set_regressor.set_regressor("invalid")
