"""Set classifier function."""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]

from tsml_eval.utils.functions import str_in_nested_list

distance_based_clusterers = [
    ["TimeSeriesKMeans", "kmeans-dtw", "k-means-dtw"],
    ["kmeans-ed", "k-means-ed"],
    ["kmeans-edr", "k-means-edr"],
    ["kmeans-erp", "k-means-erp"],
    ["kmeans-lcss", "k-means-lcss"],
    ["kmeans-msm", "k-means-msm"],
    ["kmeans-twe", "k-means-twe"],
    ["kmeans-wddtw", "k-means-wddtw"],
    ["kmeans-wdtw", "k-means-wdtw"],
    ["kmeans-ddtw", "k-means-ddtw"],
    ["TimeSeriesKMedoids", "kmedoids-dtw", "k-medoids-dtw"],
    ["kmedoids-ed", "k-medoids-ed"],
    ["kmedoids-edr", "k-medoids-edr"],
    ["kmedoids-erp", "k-medoids-erp"],
    ["kmedoids-lcss", "k-medoids-lcss"],
    ["kmedoids-msm", "k-medoids-msm"],
    ["kmedoids-twe", "k-medoids-twe"],
    ["kmedoids-wddtw", "k-medoids-wddtw"],
    ["kmedoids-wdtw", "k-medoids-wdtw"],
    ["kmedoids-ddtw", "k-medoids-ddtw"],
]
other_clusterers = [
    ["DummyClusterer", "dummy", "dummyclusterer-tsml"],
    "dummyclusterer-aeon",
    "dummyclusterer-sklearn",
]
vector_clusterers = [
    "KMeans",
]


def set_clusterer(
    clusterer_name,
    random_state=None,
    n_jobs=1,
    build_train_file=False,
    fit_contract=0,
    checkpoint=None,
    **kwargs,
):
    """Return a clusterer matching a given input name.

    Basic way of creating a clusterer to build using the default or alternative
    settings. This set up is to help with batch jobs for multiple problems and to
    facilitate easy reproducibility through run_clustering_experiment.

    Generally, inputting a clusterer class name will return said clusterer with
    default settings.

    Parameters
    ----------
    clusterer_name : str
        String indicating which clusterer to be returned.
    random_state : int, RandomState instance or None, default=None
        Random seed or RandomState object to be used in the clusterer if available.
    n_jobs: int, default=1
        The number of jobs to run in parallel for both clusterer ``fit`` and
        ``predict`` if available. `-1` means using all processors.

    Return
    ------
    clusterer: A BaseClusterer.
        The clusterer matching the input clusterer name.
    """
    c = clusterer_name.lower()

    if str_in_nested_list(distance_based_clusterers, c):
        return _set_clusterer_distance_based(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(other_clusterers, c):
        return _set_clusterer_other(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    elif str_in_nested_list(vector_clusterers, c):
        return _set_clusterer_vector(
            c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
        )
    else:
        raise ValueError(f"UNKNOWN CLUSTERER {c} in set_clusterer")


def _set_clusterer_distance_based(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    from aeon.clustering.k_means import TimeSeriesKMeans
    from aeon.clustering.k_medoids import TimeSeriesKMedoids

    if c == "timeserieskmeans" or c == "kmeans-dtw" or c == "k-means-dtw":
        return TimeSeriesKMeans(metric="dtw", random_state=random_state, **kwargs)
    if c == "kmeans-ddtw" or c == "k-means-ddtw":
        return TimeSeriesKMeans(metric="ddtw", random_state=random_state, **kwargs)
    if c == "kmeans-ed" or c == "k-means-ed":
        return TimeSeriesKMeans(metric="euclidean", random_state=random_state, **kwargs)
    if c == "kmeans-edr" or c == "k-means-edr":
        return TimeSeriesKMeans(metric="edr", random_state=random_state, **kwargs)
    if c == "kmeans-erp" or c == "k-means-erp":
        return TimeSeriesKMeans(metric="erp", random_state=random_state, **kwargs)
    if c == "kmeans-lcss" or c == "k-means-lcss":
        return TimeSeriesKMeans(metric="lcss", random_state=random_state, **kwargs)
    if c == "kmeans-msm" or c == "k-means-msm":
        return TimeSeriesKMeans(metric="msm", random_state=random_state, **kwargs)
    if c == "kmeans-twe" or c == "k-means-twe":
        return TimeSeriesKMeans(metric="twe", random_state=random_state, **kwargs)
    if c == "kmeans-wddtw" or c == "k-means-wddtw":
        return TimeSeriesKMeans(metric="wddtw", random_state=random_state, **kwargs)
    if c == "kmeans-dtw" or c == "k-means-dtw":
        return TimeSeriesKMeans(metric="dtw", random_state=random_state, **kwargs)

    elif c == "timeserieskmedoids" or c == "kmedoids-dtw" or c == "k-medoids-dtw":
        return TimeSeriesKMedoids(metric="dtw", random_state=random_state, **kwargs)


def _set_clusterer_other(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "dummyclusterer" or c == "dummy" or c == "dummyclusterer-tsml":
        from tsml.dummy import DummyClusterer

        return DummyClusterer(
            strategy="random", n_clusters=1, random_state=random_state, **kwargs
        )

    elif c == "dummyclusterer-aeon":
        from aeon.clustering.k_means import TimeSeriesKMeans

        return TimeSeriesKMeans(
            n_clusters=1,
            n_init=1,
            init_algorithm="random",
            metric="euclidean",
            max_iter=1,
            random_state=random_state,
            **kwargs,
        )
    elif c == "dummyclusterer-sklearn":
        from sklearn.cluster import KMeans

        return KMeans(
            n_clusters=1,
            n_init=1,
            init="random",
            max_iter=1,
            random_state=random_state,
            **kwargs,
        )


def _set_clusterer_vector(
    c, random_state, n_jobs, build_train_file, fit_contract, checkpoint, kwargs
):
    if c == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(random_state=random_state, **kwargs)
