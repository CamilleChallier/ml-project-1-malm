from copy import deepcopy

import numpy as np


class KMeans:
    """
    K-Means clustering algorithm Implementation
    """

    def __init__(self, n_clusters=8, max_iter=300, seed=1):
        """
        Initialize K-Means clustering algorithm

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to form as well as the number of centroids to generate.
        max_iter : int, optional
            Maximum number of iterations of the k-means algorithm for a single run.
        seed : int, optional
            Random seed.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.verbose = verbose

    def kmeans_plus(self, X_train):
        """
        K-Means++ initialisation algorithm. Do not randomly initialise the centroid, but instead choose spacially spaced centroids.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training data.

        Returns
        -------
        numpy.ndarray
            Initial centroids.
        """
        np.random.seed(self.seed)
        print(f"Initialising cluster {1}/{self.n_clusters}")
        self.centroids = [X_train[np.random.choice(X_train.shape[0], 1, replace=False)]]
        for i in range(self.n_clusters - 1):
            print(f"Initialising cluster {i+2}/{self.n_clusters}")
            # Calculate distances from points to the centroids
            dists = np.sum(
                np.array(
                    [
                        np.linalg.norm(X_train - centroid, axis=1)
                        for centroid in self.centroids
                    ]
                ),
                axis=0,
            )
            # Normalize the distances to become probabilities
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            (new_centroid_idx,) = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx].reshape(1, -1)]
        self.centroids = np.array(self.centroids)
        return self.centroids

    def fit(self, X_train):
        """
        Compute k-means clustering, Update the centroid location until the centroids no longer move or the maximum number of iterations is reached.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training data.
        """
        # Compute initial position of the centroids
        self.centroids = self.kmeans_plus(X_train)
        iteration = 0
        prev_centroids = None

        while (
            np.not_equal(self.centroids, prev_centroids).any()
            and iteration < self.max_iter
        ):
            prev_centroids = deepcopy(self.centroids)
            distances = np.array(
                [
                    np.linalg.norm(X_train - centroid, axis=1)
                    for centroid in self.centroids
                ]
            )
            centroid_idxs = np.argmin(distances, axis=0)
            self.centroids = np.array(
                [
                    np.mean(X_train[centroid_idxs == i], axis=0)
                    if (centroid_idxs == i).sum() > 0
                    else self.centroids[i]
                    for i in range(self.centroids.shape[0])
                ]
            ).squeeze()

            iteration += 1

            if iteration % 10 == 0 and self.verbose:
                print(f"Iteration: {iteration}/{self.max_iter}")
        self.last_centroid_idxs = centroid_idxs

    def evaluate(self, X):
        """
        Compute the euclidean distance between each point and its closest centroid.

        Parameters
        ----------
        X : numpy.ndarray
            Data to predict.
        """
        distances = np.array(
            [np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids]
        )
        centroid_idxs = np.argmin(distances, axis=0)
        return centroid_idxs
