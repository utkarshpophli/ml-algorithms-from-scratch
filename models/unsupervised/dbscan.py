import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.X = X
        self.labels = -1 * np.ones(X.shape[0])
        self.cluster_id = 0
        
        for i in range(X.shape[0]):
            if self.labels[i] == -1:
                if self.expand_cluster(i):
                    self.cluster_id += 1

        return self

    def expand_cluster(self, i):
        neighbors = self.region_query(i)
        if len(neighbors) < self.min_samples:
            self.labels[i] = -1
            return False
        else:
            self.labels[i] = self.cluster_id
            for neighbor in neighbors:
                if self.labels[neighbor] == -1:
                    self.labels[neighbor] = self.cluster_id
                    new_neighbors = self.region_query(neighbor)
                    if len(new_neighbors) >= self.min_samples:
                        neighbors = np.append(neighbors, new_neighbors)
            return True

    def region_query(self, i):
        neighbors = []
        for j in range(self.X.shape[0]):
            if np.linalg.norm(self.X[i] - self.X[j]) < self.eps:
                neighbors.append(j)
        return np.array(neighbors)

    def predict(self, X):
        return self.labels
