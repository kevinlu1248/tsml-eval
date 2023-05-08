import numpy as np
from aeon.distances import pairwise_distance
from aeon.clustering.base import BaseClusterer, TimeSeriesInstances
import kmedoids


class KmedoidsWrapper(BaseClusterer):

    def __init__(
            self,
            n_clusters: int,
            metric: str,
            metric_params: dict = None,
            model: str = "pam",
            init: str = "random"
    ):
        self.metric = metric
        self.init = init
        if metric_params is None:
            metric_params = {}
        self.metric_params = metric_params

        self.model = model
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

        super(KmedoidsWrapper, self).__init__(n_clusters=n_clusters)

    def _score(self, X, y=None):
        return -self.inertia_

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pairwise = pairwise_distance(
            X, self.cluster_centers_, metric=self.metric, **self.metric_params
        )
        return pairwise.argmin(axis=1)

    def _fit(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        pairwise = pairwise_distance(X, metric=self.metric, **self.metric_params)

        model = kmedoids.KMedoids(
            n_clusters=self.n_clusters,
            metric="precomputed",
            method=self.model,
            init=self.init,
        )
        model.fit(pairwise)
        self.cluster_centers_ = X[model.medoid_indices_]
        self.inertia_ = model.inertia_
        return model.labels_
