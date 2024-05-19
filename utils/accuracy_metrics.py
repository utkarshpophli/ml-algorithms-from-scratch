import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# Function to calculate Mean Absolute Error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Function to calculate Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to calculate R-squared score
def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# def accuracy_score(y_true, y_pred):
#     return np.mean(y_true == y_pred)

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    return accuracy

def precision_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives

def recall_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


def silhouette_score(X, labels):
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    distances = euclidean_distances(X)

    silhouette_scores = []

    for i in range(n_samples):
        # Calculate the average intra-cluster distance (a) for sample i
        cluster_label = labels[i]
        intra_cluster_distances = []
        for j in range(n_samples):
            if labels[j] == cluster_label and j != i:
                intra_cluster_distances.append(distances[i, j])
        a = np.mean(intra_cluster_distances)

        # Calculate the average nearest-cluster distance (b) for sample i
        nearest_cluster_distances = []
        for label in unique_labels:
            if label != cluster_label:
                other_cluster_distances = []
                for j in range(n_samples):
                    if labels[j] == label:
                        other_cluster_distances.append(distances[i, j])
                nearest_cluster_distances.append(np.mean(other_cluster_distances))
        b = min(nearest_cluster_distances)

        # Compute the silhouette score for sample i
        silhouette_i = (b - a) / max(a, b)
        silhouette_scores.append(silhouette_i)

    # Calculate the mean silhouette score across all samples
    silhouette_avg = np.mean(silhouette_scores)
    return silhouette_avg
