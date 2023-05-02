# -*- coding: utf-8 -*-
"""Shapelet transformers.

A transformer from the time domain into the shapelet domain.
"""

__author__ = ["MatthewMiddlehurst", "jasonlines", "dguijo"]
__all__ = ["RandomShapelet2Transform"]

import heapq

# import pickle
import time

import numpy as np
import pandas as pd
from aeon.transformations.base import BaseTransformer
from aeon.utils.numba.general import z_normalise_series
from aeon.utils.validation import check_n_jobs
from joblib import Parallel, delayed
from numba import njit
from numba.typed.typedlist import List
from scipy.spatial.distance import cdist

# from scipy.stats import gamma, linregress, truncnorm
from scipy.stats import linregress

# from sklearn import preprocessing
# from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

# import warnings
# from itertools import zip_longest
# from operator import itemgetter

# from sklearn.utils.multiclass import class_distribution


class RandomShapelet2Transform(BaseTransformer):
    """Random Shapelet Transform.

    Implementation of the binary shapelet transform along the lines of [1]_[2]_, with
    randomly extracted shapelets.

    Overview: Input "n" series with "d" dimensions of length "m". Continuously extract
    candidate shapelets and filter them in batches.
        For each candidate shapelet
            - Extract a shapelet from an instance with random length, position and
              dimension
            - Using its distance to train cases, calculate the shapelets information
              gain
            - Abandon evaluating the shapelet if it is impossible to obtain a higher
              information gain than the current worst
        For each shapelet batch
            - Add each candidate to its classes shapelet heap, removing the lowest
              information gain shapelet if the max number of shapelets has been met
            - Remove self-similar shapelets from the heap
    Using the final set of filtered shapelets, transform the data into a vector of
    of distances from a series to each shapelet.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to <= max_shapelets, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to n_classes / max_shapelets. If None uses the min between
        10 * n_instances and 1000
    min_shapelet_length : int, default=3
        Lower bound on candidate shapelet lengths.
    max_shapelet_length : int or None, default= None
        Upper bound on candidate shapelet lengths. If None no max length is used.
    remove_self_similar : boolean, default=True
        Remove overlapping "self-similar" shapelets when merging candidate shapelets.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_shapelet_samples.
        Default of 0 means n_shapelet_samples is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when time_limit_in_minutes is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_instances : int
        The number of train cases.
    n_dims : int
        The number of dimensions per case.
    series_length : int
        The length of each series.
    classes_ : list
        The classes labels.
    shapelets : list
        The stored shapelets and relating information after a dataset has been
        processed.
        Each item in the list is a tuple containing the following 7 items:
        (shapelet information gain, shapelet length, start position the shapelet was
        extracted from, shapelet dimension, index of the instance the shapelet was
        extracted from in fit, class value of the shapelet, The z-normalised shapelet
        array)

    See Also
    --------
    ShapeletTransformClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/transformers/ShapeletTransform.java>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge Centered
       Systems, 32, 2017.

    Examples
    --------
    >>> from aeon.transformations.panel.shapelet_transform import (
    ...     RandomShapeletTransform
    ... )
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> t = RandomShapeletTransform(
    ...     n_shapelet_samples=500,
    ...     max_shapelets=10,
    ...     batch_size=100,
    ... )
    >>> t.fit(X_train, y_train)
    RandomShapeletTransform(...)
    >>> X_t = t.transform(X_train)
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "fit_is_empty": False,
        "requires_y": True,
    }

    def __init__(
        self,
        n_shapelet_samples=10000,
        max_shapelets=None,
        min_shapelet_length=3,
        max_shapelet_length=None,
        remove_self_similar=True,
        time_limit_in_minutes=0.0,
        contract_max_n_shapelet_samples=np.inf,
        alpha=0.5,
        n_jobs=1,
        parallel_backend=None,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.remove_self_similar = remove_self_similar

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples
        self.alpha = alpha

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.batch_size = batch_size
        self.random_state = random_state

        # The following set in method fit
        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.shapelets = []

        self._n_shapelet_samples = n_shapelet_samples
        self._max_shapelets = max_shapelets
        self._max_shapelet_length = max_shapelet_length
        self._n_jobs = n_jobs
        self._batch_size = batch_size
        self._sorted_indicies = []

        super(RandomShapelet2Transform, self).__init__()

    def _fit(self, X, y=None):
        """Fit the shapelet transform to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : RandomShapeletTransform
            This estimator.
        """
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_instances, self.n_dims, self.series_length = X.shape

        if self.max_shapelets is None:
            self._max_shapelets = min(10 * self.n_instances, 1000)

        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.series_length

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        fit_time = 0

        shapelets = List([(-1.0, -1, -1, -1, -1, -1, -1.0)])
        n_shapelets_extracted = 0

        if time_limit > 0:
            while (
                fit_time < time_limit
                and n_shapelets_extracted < self.contract_max_n_shapelet_samples
            ):
                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                    )
                    for i in range(self._batch_size)
                )

                self._merge_shapelets(
                    shapelets,
                    List(candidate_shapelets),
                    self._max_shapelets,
                )

                if self.remove_self_similar:
                    to_keep = self._remove_self_similar_shapelets(shapelets)
                    shapelets = List([n for (n, b) in zip(shapelets, to_keep) if b])

                n_shapelets_extracted += self._batch_size
                fit_time = time.time() - start_time
        else:
            while n_shapelets_extracted < self._n_shapelet_samples:
                n_shapelets_to_extract = (
                    self._batch_size
                    if n_shapelets_extracted + self._batch_size
                    <= self._n_shapelet_samples
                    else self._n_shapelet_samples - n_shapelets_extracted
                )

                # Characteristics for this batch of shapelets are obtained.
                candidate_shp_characteristics = self._extract_shp_characteristics(
                    n_shapelets_extracted, n_shapelets_to_extract
                )

                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        candidate_shp_characteristics[:, i],
                    )
                    for i in range(n_shapelets_to_extract)
                )

                self._merge_shapelets(
                    shapelets, List(candidate_shapelets), self._max_shapelets
                )

                if self.remove_self_similar:
                    to_keep = self._remove_self_similar_shapelets(shapelets, self.alpha)
                    shapelets = List([n for (n, b) in zip(shapelets, to_keep) if b])

                # print(len(shapelets))

                n_shapelets_extracted += n_shapelets_to_extract

        self.shapelets = shapelets
        self.shapelets = [
            (
                s[0],  # quality
                s[1],  # length
                s[2],  # position
                s[3],  # dilation
                s[4],  # dim
                s[5],  # inst_idx
                s[6],  # y[inst_idx]
                z_normalise_series(
                    X[s[5], s[4], range(s[2], s[2] + (s[1] * s[3]), s[3])]
                ),
            )
            for s in shapelets
            if s[0] > 0
        ]

        self.shapelets.sort(
            reverse=True, key=lambda s: (s[0], s[1], s[2], s[3], s[4], s[5])
        )

        to_keep = self._remove_identical_shapelets(List(self.shapelets))
        self.shapelets = [n for (n, b) in zip(self.shapelets, to_keep) if b]

        # file = open('shapelets.p', 'wb')
        # pickle.dump(self.shapelets, file)

        self._sorted_indicies = []
        for s in self.shapelets:
            sabs = np.abs(s[7])
            self._sorted_indicies.append(
                np.array(
                    sorted(range(s[1]), reverse=True, key=lambda j, sabs=sabs: sabs[j])
                )
            )
        return self

    def _transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            The transformed dataframe in tabular format.
        """
        output = np.zeros((len(X), len(self.shapelets)))

        for i, series in enumerate(X):
            dists = Parallel(
                n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
            )(
                delayed(_online_shapelet_distance)(
                    series[s[4]][range(s[2], s[2] + (s[1] * s[3]), s[3])],
                    s[7],
                    self._sorted_indicies[n],
                    s[2],
                    s[1],
                )
                for n, s in enumerate(self.shapelets)
            )

            output[i] = dists

        return pd.DataFrame(output)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"max_shapelets": 5, "n_shapelet_samples": 50, "batch_size": 20}

    def _extract_shp_characteristics(
        self, n_shapelets_extracted, n_shapelets_to_extract
    ):
        rs = (
            None
            if self.random_state is None
            else ((self.random_state + 1) * 37 * (n_shapelets_extracted + 1))
            % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        # Indices from which shapelets are extracted.
        inst_idxs = (
            np.array(
                range(
                    n_shapelets_extracted,
                    n_shapelets_extracted + n_shapelets_to_extract,
                )
            )
            % self.n_instances
        )

        # Lengths of the shapelets of the batch. Non-uniform distribution to favour
        # short shapelets.
        samples = rng.gamma(2, scale=0.5, size=100000)
        idx = rng.randint(low=0, high=100000, size=n_shapelets_to_extract)
        samples = samples[idx]
        samples = (samples - min(samples)) / (max(samples) - min(samples))
        samples = samples * (self._max_shapelet_length - 1 - self.min_shapelet_length)
        lengths = np.floor(samples + self.min_shapelet_length).astype(int)

        # Dilations applied to the shapelets. Uniform distribution.
        upper_bounds = np.floor(
            np.log2(np.floor_divide(self.series_length - 1, lengths - 1)) + 1
        )
        dilations = np.power(
            2, rng.randint(low=0, high=upper_bounds, size=n_shapelets_to_extract)
        )

        # Positions of the shapelets of the batch. Uniform distribution.
        positions = rng.randint(
            low=0,
            high=self.series_length - ((lengths - 1) * dilations),
            size=n_shapelets_to_extract,
        )

        # Dimensions of the shapelets of the batch. Uniform distribution.
        dims = rng.randint(low=0, high=self.n_dims, size=n_shapelets_to_extract)

        return np.array((inst_idxs, lengths, positions, dims, dilations))

    def _extract_random_shapelet(self, X, y, candidate_shp_characteristics):
        inst_idx, length, position, dim, dilation = candidate_shp_characteristics

        shapelet = z_normalise_series(
            X[inst_idx, dim, range(position, position + (length * dilation), dilation)]
        )
        sabs = np.abs(shapelet)
        sorted_indicies = np.array(
            sorted(range(length), reverse=True, key=lambda j: sabs[j])
        )

        quality = self._find_shapelet_quality(
            X,
            y,
            shapelet,
            sorted_indicies,
            position,
            length,
            dim,
            inst_idx,
            dilation,
            self.series_length,
        )

        return quality, length, position, dilation, dim, inst_idx, y[inst_idx]

    @staticmethod
    def _find_shapelet_quality(
        X,
        y,
        shapelet,
        sorted_indicies,
        position,
        length,
        dim,
        inst_idx,
        dilation,
        series_length,
    ):
        # todo optimise this more, we spend 99% of time here
        orderline = []

        # This func avoids using a for loop.
        # dists = _shapelet_distance_matrix(
        #     np.array(X[:, dim, range(0, series_length, dilation)]), np.array(shapelet)
        # )

        for i, series in enumerate(X):
            if i != inst_idx:
                distance = _shapelet_distance_matrix(
                    np.array(
                        series[dim][
                            range(position, position + (length * dilation), dilation)
                        ]
                    ),
                    np.array(shapelet),
                )
                # distance = _online_shapelet_distance(
                #     series[dim][
                #       range(position, position + (length * dilation), dilation)],
                #     shapelet,
                #     sorted_indicies,
                #     position,
                #     length,
                # )
            else:
                distance = 0

            orderline.append((distance, y[i]))

        quality = calc_correlation(orderline)
        # quality = decision_tree_regressor_metric(orderline)

        return round(quality, 12)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _merge_shapelets(shapelet_heap, candidate_shapelets, max_shapelets):
        for shapelet in candidate_shapelets:
            if (
                len(shapelet_heap) == max_shapelets
                and shapelet[0] < shapelet_heap[0][0]
            ):
                continue

            # The queue is full.
            if len(shapelet_heap) > max_shapelets:
                heapq.heappushpop(shapelet_heap, shapelet)

            # The new shp is good and there is space.
            else:
                heapq.heappush(shapelet_heap, shapelet)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_self_similar_shapelets(shapelet_heap, alpha):
        to_keep = [True] * len(shapelet_heap)

        for i in range(len(shapelet_heap)):
            if not to_keep[i]:
                continue

            for n in range(i + 1, len(shapelet_heap)):
                if (
                    to_keep[n]
                    and (_is_self_similar(shapelet_heap[i], shapelet_heap[n], alpha))
                    # or np.array_equal(shapelet_heap[i][6], shapelet_heap[n][6])
                ):
                    if shapelet_heap[i][0] >= shapelet_heap[n][0]:
                        to_keep[n] = False
                    else:
                        to_keep[i] = False
                        break

        return to_keep

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_identical_shapelets(shapelets):
        to_keep = [True] * len(shapelets)

        for i in range(len(shapelets)):
            for n in range(i + 1, len(shapelets)):
                if shapelets[i][1] == shapelets[n][1] and np.array_equal(
                    shapelets[i][7], shapelets[n][7]
                ):
                    to_keep[n] = False

        return to_keep

    # print(f'{series.shape}, {shapelet.shape} {serie_sw.shape}')


# @njit(fastmath=True, cache=True)
def _shapelet_distance_matrix(
    series,
    shapelet,
):
    serie_sw = series[
        np.arange(series.size - shapelet.shape[0] + 1)[:, None]
        + np.arange(shapelet.shape[0])
    ]
    dist = cdist(shapelet.reshape(1, -1), serie_sw)
    return dist.min()


@njit(fastmath=True, cache=True)
def _online_shapelet_distance(series, shapelet, sorted_indicies, position, length):
    subseq = series[position : position + length]

    sum_ = np.sum(subseq)
    sum2 = np.sum(subseq**2)

    mean = sum_ / length
    std = np.sqrt((sum2 / length) - (mean**2))
    if std > 0:
        subseq = (subseq - mean) / std
    else:
        subseq = np.zeros(length)

    best_dist = 0
    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [sum_, sum_]
    sums2 = [sum2, sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            start = series[pos - n]
            end = series[pos - n + length]

            sums[n] += mod * end - mod * start
            sums2[n] += (mod * (end**2)) - (mod * (start**2))

            mean = sums[n] / length
            std = np.sqrt((sums2[n] / length) - (mean**2))

            dist = 0
            use_std = std != 0
            for j in range(length):
                val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist
    # return best_dist if best_dist == 0 else 1 / length * best_dist


@njit(fastmath=True, cache=True)
def _is_self_similar(s1, s2, alpha):
    # not self similar if from different series, dimension or dilation
    if (s1[5] == s2[5]) and (s1[4] == s2[4]) and (s1[3] == s2[3]):
        if s1[2] <= s2[2] <= np.floor((s1[2] + s1[1]) * alpha):
            return True

        if np.floor((s1[2] + s1[1]) * alpha) <= s2[2] + s2[1] <= s1[2] + s1[1]:
            return True

    return False


# potential metrics


def calc_correlation(orderline):
    orderline = np.array(orderline)

    if len(np.unique(orderline[:, 0])) == 1:
        return 0.0

    _, _, r_value, _, _ = linregress(orderline[:, 0], orderline[:, 1])
    return r_value**2


# def decision_tree_regressor_metric():
#     breakpoints = np.zeros((self.word_length, self.alphabet_size))
#     clf = DecisionTreeRegressor(
#         criterion="squared_error",
#         max_depth=int(np.floor(np.log2(self.alphabet_size))),
#         max_leaf_nodes=self.alphabet_size,
#         random_state=1,
#     )

#     for i in range(self.word_length):
#         clf.fit(dft[:, i][:, None], y)
#         threshold = clf.tree_.threshold[clf.tree_.children_left != -1]
#         for bp in range(len(threshold)):
#             breakpoints[i][bp] = threshold[bp]
#         for bp in range(len(threshold), self.alphabet_size):
#             breakpoints[i][bp] = sys.float_info.max

#     return np.sort(breakpoints, axis=1)
