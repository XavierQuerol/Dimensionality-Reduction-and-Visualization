import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from code.pca import customPCA
from sklearn.metrics import mean_squared_error
import time
import psutil
import os

def customPCA_analysis(X, n_components=3):
    # Record the start time and memory usage for performance tracking
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB

    # Initialize and fit PCA
    pca = customPCA()
    X_pca = pca.reduce_dim(X, n_components=n_components)

    # Record end time and memory usage for performance tracking
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB

    # Extract the principal components and explained variance
    components = pca.remaining_components[:n_components]
    explained_variance = pca.explained_variance[:n_components]

    # Reconstruct the data and calculate reconstruction error
    X_reconstructed = pca.reconstruct(X_pca)
    reconstruction_error = np.sqrt(mean_squared_error(X, X_reconstructed))

    # Performance metrics
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    pca_results = {
        'PCA Type': 'customPCA',
        'Components': [components.tolist()],
        'Explained Variance': [explained_variance.tolist()],
        'Projected Data Shape': [X_pca.shape],
        'Reconstruction Error': [reconstruction_error],
        'Time Taken (s)': [time_taken],
        'Memory Used (MB)': [memory_used]
    }
    pca_df = pd.DataFrame(pca_results)
    return pca_df

def pca_analysis(X, n_components=3):
    # Record the start time and memory usage for performance tracking
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB

    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Record end time and memory usage for performance tracking
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB

    # Extract the principal components and explained variance
    components = pca.components_[:n_components]  # The directions of the components
    explained_variance = pca.explained_variance_ratio_[:n_components]  # The proportion of variance explained

    # Reconstruct the data and calculate reconstruction error
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.sqrt(mean_squared_error(X, X_reconstructed))

    # Performance metrics
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    pca_results = {
        'PCA Type': 'PCA',
        'Components': [components.tolist()],
        'Explained Variance': [explained_variance.tolist()],
        'Projected Data Shape': [X_pca.shape],
        'Reconstruction Error': [reconstruction_error],
        'Time Taken (s)': [time_taken],
        'Memory Used (MB)': [memory_used]
    }
    pca_df = pd.DataFrame(pca_results)
    return pca_df


def incremental_pca_analysis(X, n_components=3, batch_size=10):
    # Record the start time and memory usage for performance tracking
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB

    # Initialize and fit IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_ipca = ipca.fit_transform(X)

    # Record end time and memory usage for performance tracking
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # Memory in MB

    # Extract the principal components and explained variance
    components = ipca.components_  # The directions of the components
    explained_variance = ipca.explained_variance_ratio_  # The proportion of variance explained

    # Reconstruct the data and calculate reconstruction error
    X_reconstructed = ipca.inverse_transform(X_ipca)
    reconstruction_error = np.sqrt(mean_squared_error(X, X_reconstructed))

    # Performance metrics
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    ipca_results = {
        'PCA Type': 'IncrementalPCA',
        'Components': [components.tolist()],
        'Explained Variance': [explained_variance.tolist()],
        'Projected Data Shape': [X_ipca.shape],
        'Reconstruction Error': [reconstruction_error],
        'Time Taken (s)': [time_taken],
        'Memory Used (MB)': [memory_used]
    }
    ipca_df = pd.DataFrame(ipca_results)
    return ipca_df