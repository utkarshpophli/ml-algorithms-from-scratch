
linear_regression = """class LinearRegression:
        def __init__(self, learning_rate=0.01, n_iterations=1000):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)
            self.bias = 0

            for _ in range(self.n_iterations):
                y_predicted = np.dot(X, self.weights) + self.bias
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * np.sum(y_predicted - y)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        def predict(self, X):
            return np.dot(X, self.weights) + self.bias"""

logistic_regression = """
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))"""

decision_tree = """class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, n_features, replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return self.Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = np.bincount(y)
        return np.argmax(counter)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
"""

adaboost = """class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")

            # greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred"""

knn = """class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]"""

naivebayes = """class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator"""

nn = """class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.01, n_iterations=1000):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(n_inputs, n_hidden)
        self.bias_hidden = np.zeros((1, n_hidden))
        self.weights_hidden_output = np.random.randn(n_hidden, n_outputs)
        self.bias_output = np.zeros((1, n_outputs))
    
    def fit(self, X, y):
        for _ in range(self.n_iterations):
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self._sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            predictions = self._sigmoid(output_layer_input)
            
            # Backward pass
            error = predictions - y
            d_output = error * self._sigmoid_derivative(predictions)
            
            error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self._sigmoid_derivative(hidden_layer_output)
            
            # Update weights and biases
            self.weights_hidden_output -= hidden_layer_output.T.dot(d_output) * self.learning_rate
            self.bias_output -= np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.weights_input_hidden -= X.T.dot(d_hidden_layer) * self.learning_rate
            self.bias_hidden -= np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate
    
    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self._sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        predictions = self._sigmoid(output_layer_input)
        return predictions
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
"""

randomForest = """def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)"""

svm = """class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)"""

apriori = """class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.support_data = {}

    def fit(self, transactions):
        self.transactions = list(map(set, transactions))
        self.num_transactions = len(transactions)
        self.itemsets = self.create_initial_itemsets()
        self.frequent_itemsets = [self.find_frequent_itemsets(self.itemsets)]

        k = 2
        while True:
            candidate_itemsets = self.apriori_gen(self.frequent_itemsets[-1], k)
            frequent_itemsets_k = self.find_frequent_itemsets(candidate_itemsets)
            if not frequent_itemsets_k:
                break
            self.frequent_itemsets.append(frequent_itemsets_k)
            k += 1

        self.generate_association_rules()

    def create_initial_itemsets(self):
        itemsets = []
        for transaction in self.transactions:
            for item in transaction:
                if frozenset([item]) not in itemsets:
                    itemsets.append(frozenset([item]))
        itemsets.sort()
        return itemsets

    def find_frequent_itemsets(self, itemsets):
        itemset_counts = {}
        for transaction in self.transactions:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    if itemset not in itemset_counts:
                        itemset_counts[itemset] = 1
                    else:
                        itemset_counts[itemset] += 1

        num_transactions = float(len(self.transactions))
        frequent_itemsets = []
        for itemset, count in itemset_counts.items():
            support = count / num_transactions
            if support >= self.min_support:
                frequent_itemsets.append(itemset)
                self.support_data[itemset] = support
        return frequent_itemsets

    def apriori_gen(self, itemsets, k):
        candidates = []
        len_itemsets = len(itemsets)
        for i in range(len_itemsets):
            for j in range(i + 1, len_itemsets):
                L1 = list(itemsets[i])[:k-2]
                L2 = list(itemsets[j])[:k-2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    candidates.append(itemsets[i] | itemsets[j])
        return candidates

    def generate_association_rules(self):
        self.rules = []
        for itemsets in self.frequent_itemsets[1:]:
            for freq_set in itemsets:
                H1 = [frozenset([item]) for item in freq_set]
                if len(freq_set) > 1:
                    self.rules_from_conseq(freq_set, H1)

    def rules_from_conseq(self, freq_set, H):
        m = len(H[0])
        if len(freq_set) > (m + 1):
            Hmp1 = self.apriori_gen(H, m + 1)
            Hmp1 = self.calc_confidence(freq_set, Hmp1)
            if Hmp1:
                self.rules_from_conseq(freq_set, Hmp1)

    def calc_confidence(self, freq_set, H):
        pruned_H = []
        for conseq in H:
            if freq_set - conseq in self.support_data:
                conf = self.support_data[freq_set] / self.support_data[freq_set - conseq]
                if conf >= self.min_confidence:
                    self.rules.append((list(freq_set - conseq), list(conseq), conf))  # Fix here
                    pruned_H.append(conseq)
            else:
                print(f"Missing support data for {freq_set - conseq}")
        return pruned_H


    def get_frequent_itemsets(self):
        return [(list(itemset), self.support_data[itemset]) for itemsets in self.frequent_itemsets for itemset in itemsets]

    def get_rules(self):
        return [(list(rule[0]), list(rule[1]), rule[2]) for rule in self.rules]
"""

dbscan = """
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
"""

hclustering = """class HierarchicalClustering:
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
"""

kmeans = """def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0
"""

pca = """class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)"""

perceptron = """class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.where(linear_output >= 0, 1, -1)

                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.where(linear_output >= 0, 1, -1)
        return y_predicted"""

linear = """def linear(x):
    return x """

sigmoid = """def sigmoid(x):
    return 1 / (1 + np.exp(-x))"""

relu = """def relu(x):
    return np.maximum(0, x)"""

tanh = """def tanh(x):
    return np.tanh(x)"""

softmax = """def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0))
    return exp_x / exp_x.sum(axis=0)"""

