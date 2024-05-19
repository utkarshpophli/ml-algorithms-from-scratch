import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import (make_blobs, make_classification, make_moons,
                              make_regression)

import utils.display_code as dc
from models.deep_learning.activation_function import (linear, relu, sigmoid,
                                                      softmax, tanh)
from models.deep_learning.neural_network import NeuralNetwork
from models.deep_learning.perceptron import Perceptron
from models.supervised.adaboost import Adaboost
from models.supervised.decision_tree import DecisionTree
from models.supervised.KNN import KNN
from models.supervised.linear_regression import LinearRegression
from models.supervised.logistic_regression import LogisticRegression
from models.supervised.naivebayes import NaiveBayes
from models.supervised.random_forest import RandomForest
from models.supervised.support_vector_machine import SVM
from models.unsupervised.apriori import Apriori
from models.unsupervised.dbscan import DBSCAN
from models.unsupervised.hierarchical_clustering import HierarchicalClustering
from models.unsupervised.kmeans import KMeans
from models.unsupervised.pca import PCA
from utils.accuracy_metrics import (accuracy_score, f1_score,
                                    mean_absolute_error, mean_squared_error,
                                    precision_score, r2_score, recall_score,
                                    silhouette_score)


def main():
    st.title("Machine Learning Algorithms from Scratch")
    
    # Sidebar for algorithm selection
    algorithm = st.sidebar.selectbox("Choose an algorithm", 
                                     ("Linear Regression", "Logistic Regression", "Decision Tree", 
                                      "Random Forest", "SVM", "K-Nearest Neighbors", "Naive Bayes", 
                                      "Neural Network", "K-Means Clustering", "Hierarchical Clustering", 
                                      "PCA", "AdaBoost","DBSCAN","Perceptron", "Activation Function",
                                      ))
    
    # Sidebar for parameters
    if algorithm == "Linear Regression":
        st.header('Linear Regression')
        st.subheader('Code')
        st.code(dc.linear_regression, language='py')
        
 
        lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
        n_iter = st.sidebar.slider("Number of Iterations", 100, 10000, 1000)
        
        # Generate synthetic data for demonstration
        X, y = make_regression(n_samples=100, n_features=1, noise=20)
        model = LinearRegression(learning_rate=lr, n_iterations=n_iter)
        model.fit(X, y)
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        st.subheader('Accuracy Metrics')
        st.write(f'Mean Absolute Error: {mae:.2f}')
        st.write(f'Mean Squared Error: {mse:.2f}')
        st.write(f'R-squared (R2) Score: {r2:.2f}')
        st.subheader('Visualization')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers',marker=dict(color='red'), name='Data'))
        fig.add_trace(go.Scatter(x=X.flatten(), y=predictions, mode='lines', marker=dict(color='blue'),name='Regression Line'))
        
        st.plotly_chart(fig)


    elif algorithm == "Logistic Regression":
        lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
        n_iter = st.sidebar.slider("Number of Iterations", 100, 10000, 1000)
        st.header("Logistic Regression")
        st.subheader("Code")
        st.code(dc.logistic_regression, language='py')
        
        # Generate synthetic data for demonstration with two features
        X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)
        
        model = LogisticRegression(learning_rate=lr, n_iterations=n_iter)
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Split data into two classes for coloring
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        
        # Display accuracy metrics
        st.subheader('Accuracy Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')
        
        # Create 3D scatter plot with different colors for different classes and a 3D surface plot for the decision boundary
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=class_0[:, 0], y=class_0[:, 1], z=predictions[y == 0], mode='markers', marker=dict(color='blue', symbol='circle'), name='Class 0'))
        fig.add_trace(go.Scatter3d(x=class_1[:, 0], y=class_1[:, 1], z=predictions[y == 1], mode='markers', marker=dict(color='red', symbol='x'), name='Class 1'))
        
        # Add decision boundary surface plot
        fig.add_trace(go.Surface(x=X[:, 0], y=X[:, 1], z=predictions.reshape(X[:, 0].shape), colorscale='RdBu', opacity=0.5, name='Decision Boundary'))
        st.subheader('Visualization')
        fig.update_layout(title='Visualization of Logistic Regression', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Predictions'))
        st.plotly_chart(fig)



    elif algorithm == "Decision Tree":
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 2)
        st.header("Decision Tree")
        st.subheader("Code")
        st.code(dc.decision_tree, language='py')

        
        # Generate synthetic data with two features for demonstration
        X, y = make_classification(
        n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42
    )

        # Create and train the Decision Tree model
        model = DecisionTree(min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X, y)

        # Generate a grid of points to make predictions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1), np.arange(z_min, z_max, 0.1))
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        # Get predictions from the Decision Tree model

        predictions = model.predict(X) 

        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        # Display accuracy metrics
        st.subheader('Accuracy Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')

        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=y,  # Color by original class label
                colorscale='Viridis',
                opacity=0.8
            )
        ))
        st.subheader("Visualization")
        fig.update_layout(title='Decision Tree Decision Boundaries', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))
        
        st.plotly_chart(fig)
    
    elif algorithm == "Random Forest":
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 2)
        st.header("Random Forest")
        st.subheader("Code")
        st.code(dc.randomForest, language='py')
        
        # Generate synthetic data for demonstration
        X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)
        
        # Create and train the Random Forest model
        model = RandomForest(min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X, y)
        predictions = model.predict(X)

        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        # Display accuracy metrics
        st.subheader('Accuracy Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')
        
        # Generate a grid of points to make predictions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Get predictions from each tree in the forest
        tree_predictions = []
        for tree in model.trees:
            tree_predictions.append(tree.predict(grid_points))
        
        # Aggregate predictions from all trees
        aggregated_predictions = np.mean(tree_predictions, axis=0)
        aggregated_predictions = aggregated_predictions.reshape(xx.shape)
        
        # Create 3D plot
        fig = go.Figure()
        fig.add_trace(go.Surface(x=xx, y=yy, z=aggregated_predictions, colorscale='Viridis', opacity=0.8, showscale=False))
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode='markers', marker=dict(color='blue', size=5), name='Original Data'))
        st.subheader("Visualization")
        fig.update_layout(title='Random Forest Predictions', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Predictions'))
        
        st.plotly_chart(fig)
    
    elif algorithm == "SVM":
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
        lambda_param = st.sidebar.slider("Lambda Parameter", 0.001, 0.1, 0.01)
        n_iters = st.sidebar.slider("Number of Iterations", 100, 2000, 1000)
        st.header("SVM")
        st.subheader("Code")
        st.code(dc.svm, language='py')
        
        # Generate synthetic data for demonstration with two features
        X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)
        
        # Create and train the SVM model
        model = SVM(learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
        model.fit(X, y)
        predictions = model.predict(X)

        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        # Display accuracy metrics
        st.subheader('Accuracy Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')
        
        # Create meshgrid for plotting decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Get predictions from the SVM model
        predictions = model.predict(grid_points)
        predictions = predictions.reshape(xx.shape)
        
        # Create 3D plot
        fig = go.Figure()
        fig.add_trace(go.Surface(x=xx, y=yy, z=predictions, colorscale='Viridis', opacity=0.8, showscale=False))
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode='markers', marker=dict(color='blue', size=5), name='Original Data'))
        st.subheader("Visualization")
        fig.update_layout(title='Support Vector Machine Decision Boundary', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Predictions'))
        
        st.plotly_chart(fig)
    
    elif algorithm == "K-Nearest Neighbors":
        k = st.sidebar.slider("Number of Neighbors (k)", 1, 10, 3)
        st.header("K-Nearest Neighbors")
        st.subheader("Code")
        st.code(dc.knn, language='py')
        
        # Generate synthetic data for demonstration with three features
        X, y = make_classification(n_samples=50, n_features=3, n_informative=3, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)

        # Create and train the KNN model
        model = KNN(k=k)
        model.fit(X, y)
        predictions = model.predict(X)

        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        # Display accuracy metrics
        st.subheader('Accuracy Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')
        
        # Get predictions from the KNN model
        predictions = model.predict(X)
        
        # Create 3D scatter plot for data points colored by class labels
        fig = go.Figure()
        for label in np.unique(y):
            indices = np.where(y == label)
            fig.add_trace(go.Scatter3d(
                x=X[indices, 0][0], 
                y=X[indices, 1][0], 
                z=X[indices, 2][0], 
                mode='markers', 
                marker=dict(size=8), 
                name=f'Class {label}'
            ))

        # Add predictions to the plot
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], 
            y=X[:, 1], 
            z=X[:, 2], 
            mode='markers', 
            marker=dict(size=4, color=predictions, colorscale='Viridis', opacity=0.6), 
            name='Predictions'
        ))
        st.subheader("Visualization")
        # Update layout
        fig.update_layout(title='K-Nearest Neighbors Data Points', scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3'
        ))

        # Show plot
        st.plotly_chart(fig)
    
    elif algorithm == "Naive Bayes":
    # Generate synthetic data for demonstration with two features
        X, y = make_classification(
            n_samples=50, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42
        )
        st.header("Naive Bayes")
        st.subheader("Code")
        st.code(dc.naivebayes, language='py')

        # Create and train the Naive Bayes model
        model = NaiveBayes()
        model.fit(X, y)
        predictions = model.predict(X)

        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        # Display accuracy metrics
        st.subheader('Accuracy Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')

        # Create meshgrid for plotting decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Get predictions from the Naive Bayes model
        predictions = model.predict(grid_points)
        predictions = predictions.reshape(xx.shape)

        # Create 3D plot
        fig = go.Figure()
        fig.add_trace(go.Surface(x=xx, y=yy, z=predictions, colorscale='Viridis', opacity=0.8, showscale=False))
        fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode='markers', marker=dict(color=y, colorscale='Viridis', size=5), name='Original Data'))
        st.subheader("Visualization")
        fig.update_layout(title='Naive Bayes Decision Surface', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Predictions'))

        st.plotly_chart(fig)
    
    elif algorithm == "K-Means Clustering":
    # Sidebar parameters
        K = st.sidebar.slider("Number of Clusters (K)", 2, 10, 5)
        max_iters = st.sidebar.slider("Maximum Iterations", 50, 500, 100)
        plot_steps = st.sidebar.checkbox("Plot Steps", False)
        st.header("K-Means Clustering")
        st.subheader("Code")
        st.code(dc.kmeans, language='py')

        # Generate synthetic data for demonstration with three features
        X, y = make_classification(
            n_samples=50, n_features=3, n_informative=3, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42
        )

        # Create and train the KMeans model
        kmeans = KMeans(K=K, max_iters=max_iters, plot_steps=plot_steps)
        cluster_labels = kmeans.predict(X)

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=cluster_labels,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        st.subheader("Visualization")
        fig.update_layout(title='KMeans Clustering ', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))

        st.plotly_chart(fig)

    elif algorithm == "PCA":
    # Sidebar parameter for number of components
        n_components = st.sidebar.slider("Number of Components", 2, 3, 2)
        st.header("Principal Component Analysis")
        st.subheader("Code")
        st.code(dc.pca, language='py')

        # Generate synthetic data for demonstration with the specified number of features/components
        X, y = make_classification(
            n_samples=50, n_features=n_components, n_informative=n_components, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42
        )

        # Create and train the PCA model
        pca = PCA(n_components=n_components)
        pca.fit(X)
        transformed_data = pca.transform(X)

        classifier = LogisticRegression()
        classifier.fit(transformed_data, y)
        predictions = classifier.predict(transformed_data)

        # Calculate accuracy metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        # Display accuracy metrics
        st.subheader('Accuracy Metrics')
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1 Score: {f1:.2f}')

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=transformed_data[:, 0],
            y=transformed_data[:, 1],
            z=transformed_data[:, 2] if n_components == 3 else np.zeros_like(transformed_data[:, 0]),  # Use third component if n_components is 3
            mode='markers',
            marker=dict(
                size=8,
                color=y,  # Color by class label
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        st.subheader("Visualization")

        fig.update_layout(title='PCA Projection', scene=dict(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', zaxis_title='Principal Component 3' if n_components == 3 else 'Zero'))

        st.plotly_chart(fig)
    
    elif algorithm == "Hierarchical Clustering":
    # Sidebar parameter for number of clusters
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 2)
        st.header("Hierarchical Clustering")
        st.subheader("Code")
        st.code(dc.hclustering, language='py')

        # Generate synthetic data for demonstration
        X, y = make_classification(
            n_samples=50, n_features=3, n_informative=3, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42
        )

        # Create and train the Hierarchical Clustering model
        hc = HierarchicalClustering(n_clusters=n_clusters)
        hc.fit(X)
        labels = hc.predict(X)

        silhouette_avg = silhouette_score(X, labels)

        # Display silhouette score
        st.subheader('Silhouette Score')
        st.write(f'Silhouette Score: {silhouette_avg:.2f}')
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=labels,  # Color by cluster label
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        st.subheader("Visualization")
        fig.update_layout(title='Hierarchical Clustering', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))

        st.plotly_chart(fig)

    elif algorithm == "AdaBoost":
        n_clf = st.sidebar.slider("Number of Classifiers", min_value=1, max_value=10, value=5)
        # Generate synthetic data for demonstration
        X, y = make_classification(
            n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_classes=2, n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42
        )
        st.header("AdaBoost")
        st.subheader("Code")
        st.code(dc.adaboost, language='py')

        # Create and train the Adaboost model
        model = Adaboost(n_clf=n_clf)
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)

        # Display accuracy
        st.subheader('Accuracy')
        st.write(f'Accuracy: {accuracy:.2f}')

        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=y,  # Color by original class label
                colorscale='Viridis',
                opacity=0.8
            )
        ))
        st.subheader("Visualization")
        fig.update_layout(title='Adaboost Decision Boundaries', scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))

        st.plotly_chart(fig)
    
    # elif algorithm == "Neural Network":
    #     n_hidden = st.sidebar.slider("Number of Hidden Units", 1, 100, 10)
    #     lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    #     n_iter = st.sidebar.slider("Number of Iterations", 100, 10000, 1000)
    #     st.subheader("Code")
    #     st.code(dc.nn, language='py')
        
    #     # Generate synthetic data for demonstration
    #     X, y = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, random_state=42)
    #     y = y.reshape(-1, 1)  # Reshape y for the neural network
    #     model = NeuralNetwork(n_inputs=3, n_hidden=n_hidden, n_outputs=1, learning_rate=lr, n_iterations=n_iter)
    #     model.fit(X, y)
    #     predictions = model.predict(X)
        
    #     predictions = (predictions > 0.5).astype(int)

    #     # Calculate accuracy
    #     accuracy = accuracy_score(y, predictions)

    #     # Display accuracy
    #     st.subheader('Accuracy')
    #     st.write(f'Accuracy: {accuracy:.2f}')
    #     # Binarize predictions for plotting
    #     predictions = (predictions > 0.5).astype(int)
    #     st.subheader("Visualization")
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(color=y.flatten()), name='Data'))
    #     fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(symbol='x', color=predictions.flatten()), name='Predictions'))
        
    #     st.plotly_chart(fig)
    
    elif algorithm == "DBSCAN":
        eps = st.sidebar.slider("Epsilon", 0.1, 1.0, 0.5)
        min_samples = st.sidebar.slider("Minimum Samples", 1, 10, 5)
        st.header("DBSCAN")
        st.subheader("Code")
        st.code(dc.dbscan, language='py')

        # Generate synthetic data for demonstration
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, n_features=3, random_state=0)

        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        labels = dbscan.predict(X)
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, labels)

        # Display silhouette score
        st.subheader('Silhouette Score')
        st.write(f'Silhouette Score: {silhouette_avg:.2f}')

        st.subheader("Visualization")
        # Plot results
        fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels.astype(str), title="DBSCAN Clustering")
        st.plotly_chart(fig)
    
    elif algorithm == "Apriori":
        def load_data():
            transactions = [
                ['milk', 'bread', 'butter'],
                ['beer', 'bread'],
                ['milk', 'bread', 'butter', 'beer'],
                ['bread', 'butter'],
                ['milk', 'bread', 'beer']
            ]
            return transactions
        
        min_support = st.sidebar.slider("Minimum Support", 0.01, 1.0, 0.5)
        min_confidence = st.sidebar.slider("Minimum Confidence", 0.01, 1.0, 0.7)

        st.subheader("Code")
        st.code(dc.apriori, language='py')

        transactions = load_data()
        apriori = Apriori(min_support=min_support, min_confidence=min_confidence)
        apriori.fit(transactions)
        
        frequent_itemsets = apriori.get_frequent_itemsets()
        rules = apriori.get_rules()

        st.subheader("Frequent Itemsets")
        df_itemsets = pd.DataFrame(frequent_itemsets, columns=["Itemset", "Support"])
        st.write(df_itemsets)

        st.subheader("Association Rules")
        df_rules = pd.DataFrame(rules, columns=["Antecedent", "Consequent", "Confidence"])
        st.write(df_rules)

        if df_itemsets.empty:
            st.warning("No frequent itemsets found. Try lowering the support threshold.")

        if df_rules.empty:
            st.warning("No association rules found. Try lowering the confidence threshold.")

        # Visualize association rules as a network plot
        G = nx.DiGraph()
        for rule in rules:
            antecedent, consequent, confidence = rule
            G.add_edge(antecedent, consequent, weight=confidence)
        st.subheader("Visualization")
        fig = plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, edge_color="black", linewidths=1, arrows=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        st.pyplot(fig)

    elif algorithm == "Perceptron":
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
        n_iters = st.sidebar.slider("Number of Iterations", 100, 5000, 1000)
        st.header("Perceptron")
        st.subheader("Code")
        st.code(dc.perceptron, language='py')

        # Generate synthetic data for demonstration
        X, y = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, random_state=42)
        
        # Train Perceptron
        perceptron = Perceptron(learning_rate=learning_rate, n_iters=n_iters)
        perceptron.fit(X, y)
        y_pred = perceptron.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        st.subheader("Accuracy")
        st.write(f'Accuracy: {accuracy:.2f}')

        st.subheader("Model Parameters")
        st.write(f"Weights: {perceptron.weights}")
        st.write(f"Bias: {perceptron.bias}")

        st.subheader("Visualization")
        # Plot results
        fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y.astype(str), title="Perceptron Classification")
        fig.add_traces(px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y_pred.astype(str)).data)
        fig.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        st.plotly_chart(fig)

        
    
    elif algorithm == "Activation Function":

        def plot_activation_function(name, x, y):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))

            fig.update_layout(
                title=f"{name} Activation Function",
                xaxis_title="X",
                yaxis_title="Y"
            )
            st.plotly_chart(fig)

        st.sidebar.subheader("Parameters")
        min_val = st.sidebar.number_input("Min Value", value=-5)
        max_val = st.sidebar.number_input("Max Value", value=5)
        num_points = st.sidebar.slider("Number of Points", min_value=10, max_value=1000, value=100, step=10)

        # Generate 2D data for visualization
        x = np.linspace(min_val, max_val, num_points)

        st.subheader("Linear")
        # Linear Activation Function
        y_linear = linear(x)
        plot_activation_function("Linear", x, y_linear)

        st.subheader("Sigmoid")
        # Sigmoid Activation Function
        y_sigmoid = sigmoid(x)
        plot_activation_function("Sigmoid", x, y_sigmoid)

        st.subheader("ReLU")
        # ReLU Activation Function
        y_relu = relu(x)
        plot_activation_function("ReLU", x, y_relu)

        st.subheader("Tanh")
        # Tanh Activation Function
        y_tanh = tanh(x)
        plot_activation_function("Tanh", x, y_tanh)

        st.subheader("Softmax")
        # Softmax Activation Function
        y_softmax = softmax(x)
        plot_activation_function("Softmax", x, y_softmax)
    
    elif algorithm == "Neural Network":
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
        epochs = st.sidebar.slider("Number of Epochs", 100, 5000, 1000)
        hidden_size = st.sidebar.slider("Hidden Layer Size", 1, 10, 5)
        st.header("Neural Network")
        st.subheader("Code")
        st.code(dc.nn, language='py')

        # Generate synthetic data
        X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
        y = y.reshape(-1, 1)

        # Initialize and train neural network
        nn = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
        nn.train(X, y, learning_rate, epochs)

        # Make predictions
        y_pred = nn.forward(X)
        acc = nn.accuracy(X, y)
        st.subheader("Model Accuracy")
        st.write(f"Accuracy: {acc * 100:.2f}%")

        # Prepare 3D plot
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=y.flatten(), mode='markers', marker=dict(color='blue', size=5),
            name='Actual Labels'
        ))
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=y_pred.flatten(), mode='markers', marker=dict(color='red', size=5),
            name='Predicted Labels'
        ))
        fig.update_layout(title="Neural Network Classification", scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Output'
        ))
        st.subheader("Visualization")
        st.plotly_chart(fig)

        st.subheader("Model Parameters")
        st.write(f"Weights (Input to Hidden): {nn.weights_input_hidden}")
        st.write(f"Bias (Hidden): {nn.bias_hidden}")
        st.write(f"Weights (Hidden to Output): {nn.weights_hidden_output}")
        st.write(f"Bias (Output): {nn.bias_output}")


footer = """
---
&copy; 2024 Utkarsh Pophli. All rights reserved. | [@utkarshpophli](https://github.com/utkarshpophli)
"""
st.write(footer, unsafe_allow_html=True)




if __name__ == "__main__":
    main()
