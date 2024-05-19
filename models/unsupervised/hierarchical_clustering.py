import numpy as np


class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.clusters = None
        self.X = None

    def fit(self, X):
        self.X = X
        # Initialize each data point to its own cluster
        self.clusters = [[i] for i in range(X.shape[0])]

        # Continue until we have the desired number of clusters
        while len(self.clusters) > self.n_clusters:
            # Compute pairwise distances between clusters
            distances = [[self._distance(i, j) for j in self.clusters] for i in self.clusters]

            # Find the two closest clusters
            i, j = np.unravel_index(np.argmin(distances), (len(self.clusters), len(self.clusters)))

            # Merge the two closest clusters
            self.clusters[i].extend(self.clusters[j])
            del self.clusters[j]

    def predict(self, X):
        # Assign each data point to the cluster it belongs to
        labels = np.empty(X.shape[0], dtype=np.int64)
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                labels[j] = i
        return labels

    def _distance(self, cluster1, cluster2):
        # Compute the distance between two clusters (single linkage)
        return min(np.linalg.norm(self.X[i]-self.X[j]) for i in cluster1 for j in cluster2)
