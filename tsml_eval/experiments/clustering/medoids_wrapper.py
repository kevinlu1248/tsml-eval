import math

import numpy as np
from aeon.distances import pairwise_distance
from aeon.clustering.base import BaseClusterer, TimeSeriesInstances
import random
import kmedoids
from kmedoids import pam_build
from banditpam import KMedoids as BanditPAM


kmedoids_package = ["pam", "fasterpam", "fastpam1", "pam", "fastmsc", "fastermsc",
                    "pamsil", "pammedsil", "alternate"]


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
        if self.model in kmedoids_package:
            return self._fit_kmedoids_pacakge(X)
        elif self.model == "clara":
            return self._fit_clara(X)
        elif self.model == "clarans":
            return self._fit_clarans(X)
        elif self.model == "banditpam":
            raise NotImplementedError("banditpam not implemented")
        elif self.model == "features":
            raise NotImplementedError("features not implemented")
        elif self.model == "dimreduction":
            raise NotImplementedError("dimreduction not implemented")
        else:
            raise ValueError("model invalid")

    def _fit_kmedoids_pacakge(self, X: np.ndarray, model_str=None):
        pairwise = pairwise_distance(X, metric=self.metric, **self.metric_params)
        if model_str is None:
            model_str = self.model
        model = kmedoids.KMedoids(
            n_clusters=self.n_clusters,
            metric="precomputed",
            method=model_str,
            init=self.init,
        )
        model.fit(pairwise)
        self.cluster_centers_ = X[model.medoid_indices_]
        self.inertia_ = model.inertia_
        return model.labels_

    def _fit_clara(self, X: np.ndarray):
        num_cases_to_optimise = 40 + 2 * self.n_clusters
        if num_cases_to_optimise > X.shape[0]:
            num_cases_to_optimise = X.shape[0]
        subset = X[np.random.choice(X.shape[0], num_cases_to_optimise, replace=False)]
        return self._fit_kmedoids_pacakge(subset, model_str="pam")

    def _fit_clarans(self, X: np.ndarray):
        pairwise = pairwise_distance(X, metric=self.metric, **self.metric_params)
        max_neighbours = math.ceil((1.25 / 100) * (self.n_clusters * (X.shape[0] - self.n_clusters)))
        n_init = 10
        min_cost = np.inf
        best_medoids = None

        for _ in range(n_init):

            if self.init == "random":
                medoids = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            else:
                medoids = pam_build(pairwise, self.n_clusters)

            current_cost = pairwise[medoids].min(axis=0).sum()

            j = 0
            while j < max_neighbours:
                new_medoids = medoids.copy()
                to_replace = random.randrange(self.n_clusters)
                replace_with = new_medoids[0]
                # select medoids not in new_medoids
                while replace_with in new_medoids:
                    replace_with = random.randrange(X.shape[0]) - 1
                new_medoids[to_replace] = replace_with
                new_cost = pairwise[new_medoids].min(axis=0).sum()
                if new_cost < current_cost:
                    current_cost = new_cost
                    medoids = new_medoids
                else:
                    j += 1

            if current_cost < min_cost:
                min_cost = current_cost
                best_medoids = medoids

        self.cluster_centers_ = X[best_medoids]
        self.inertia_ = min_cost
        return pairwise[best_medoids].argmin(axis=1)

    def _fit_bandit_pam(self, X: np.ndarray):
        model = BanditPAM(
            n_clusters=self.n_clusters,
            metric=self.metric,
            method=self.model,
            init=self.init,
        )
        model.fit(X)
        self.cluster_centers_ = X[model.medoid_indices_]
        self.inertia_ = model.inertia_
        return model.labels_

# from aeon.datasets import load_gunpoint
#
# if __name__ == "__main__":
#     train_X, train_y = load_gunpoint(return_X_y=True, split="train")
#     test_X, test_y = load_gunpoint(return_X_y=True, split="test")
#     model = KmedoidsWrapper(
#         n_clusters=len(set(train_y)),
#         metric="squared",
#         model="clarans",
#         init="random"
#     )
#     train_res = model.fit(train_X)
#     test_res = model.predict(test_X)
#     # print(model.cluster_centers_)
#     print(model.inertia_)
#     joe = ""
