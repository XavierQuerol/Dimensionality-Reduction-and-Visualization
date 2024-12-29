import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from code.pca import customPCA
from code.optics import apply_optics
from code.metrics import get_metrics_general
from code.global_fastkmeans import run_global_kmeans
from sklearn.decomposition import KernelPCA

def pca_cluster(data, Y, component, method='G-Means', max_clusters_gkmeans=12, optics_metric='euclidean', optics_algorithm='auto'):
    results = []
    ncol = data.shape[1]
    print(ncol)

    # Apply PCA
    pca = customPCA()
    reduced_data = pca.reduce_dim(data, n_components= (ncol+1))

    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(cumulative_explained_variance) + 1),
             cumulative_explained_variance * 100,
             marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance by Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.grid(True)
    plt.show()



    reduced_data = pca.reduce_dim(data, n_components=component)


    if method == 'G-Means':
        start_time = time.time()
        #clusters_gkmeans, labels_gkmeans = run_global_kmeans(reduced_data, max_clusters=max_clusters_gkmeans, distance= 'euclidean')
        clusters_gkmeans, labels_gkmeans = run_global_kmeans(reduced_data, max_clusters=max_clusters_gkmeans, distance= 'euclidean')
        end_time = time.time()
        g_time = end_time - start_time
        reconstructed_data = pca.reconstruct(reduced_data)

        # Calculate reconstruction error
        reconstruction_error = pca.reconstruction_error(data)

        results_means = get_metrics_general(reduced_data, Y, labels_gkmeans, method, g_time, n_iterations=None)
        results_means['n_components'] = component
        results_means['explained_variance'] = cumulative_explained_variance[component - 1]
        results_means['reconstruction_error'] = reconstruction_error
        results.append(results_means)
        print(component)


    elif method == 'Optics':
        start_time = time.time()
        optics_labels = apply_optics(reduced_data, metric=optics_metric, algorithm=optics_algorithm)
        end_time = time.time()
        o_time = end_time - start_time
        reconstructed_data = pca.reconstruct(reduced_data)

        # Calculate reconstruction error
        reconstruction_error = pca.reconstruction_error(data)

        results_optics = get_metrics_general(reduced_data, Y, optics_labels, method, o_time, n_iterations=None)
        results_optics['n_components'] = component
        results_optics['explained_variance'] = cumulative_explained_variance[component - 1]
        results_optics['reconstruction_error'] = reconstruction_error
        results.append(results_optics)

    results_df = pd.DataFrame(results)

    columns = ['n_components'] + ['explained_variance'] + ['reconstruction_error'] + [col for col in results_df.columns if col not in ['n_components', 'explained_variance', 'reconstruction_error']]
    results_df = results_df[columns]


    return results_df


def kernel_cluster(data, Y, component,  method='G-Means', max_clusters_gkmeans=12, optics_metric='euclidean', optics_algorithm='auto'):
    results = []
    ncol = data.shape[1]
    # Initialize KernelPCA
    kpca = KernelPCA(kernel='rbf', n_components=(ncol), fit_inverse_transform=True)
    reduced_data = kpca.fit_transform(data)

    # Calculate explained variance
    eigenvalues = kpca.eigenvalues_
    total_variance = np.sum(eigenvalues)
    explained_variance_ratios = eigenvalues / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratios)

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(1, len(cumulative_explained_variance) + 1),
        cumulative_explained_variance * 100,  # Percentage
        marker='o', linestyle='-'
    )
    plt.title('Cumulative Explained Variance by Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.grid(True)
    plt.show()


    # Reduce data to the current number of components
    kpca = KernelPCA(kernel='rbf', n_components=component)
    reduced_data = kpca.fit_transform(data)

    if method == 'G-Means':
        start_time = time.time()
        clusters_gkmeans, labels_gkmeans = run_global_kmeans(
            reduced_data, max_clusters=max_clusters_gkmeans, distance='euclidean'
        )
        end_time = time.time()
        g_time = end_time - start_time

        # reconstruction_error
        reconstruction_error = np.nan

        results_means = get_metrics_general(reduced_data, Y, labels_gkmeans, method, g_time, n_iterations=None)
        results_means['n_components'] = component
        results_means['explained_variance'] = cumulative_explained_variance[component - 1]
        results_means['reconstruction_error'] = reconstruction_error
        results.append(results_means)
        print(component)

    elif method == 'Optics':
        start_time = time.time()
        optics_labels = apply_optics(reduced_data, metric=optics_metric, algorithm=optics_algorithm)
        end_time = time.time()
        o_time = end_time - start_time

        # reconstruction_error
        reconstruction_error = np.nan

        results_optics = get_metrics_general(reduced_data, Y, optics_labels, method, o_time, n_iterations=None)
        results_optics['n_components'] = component
        results_optics['explained_variance'] = cumulative_explained_variance[component - 1]
        results_optics['reconstruction_error'] = reconstruction_error
        results.append(results_optics)

    # Create a DataFrame with results
    results_df = pd.DataFrame(results)
    columns = ['n_components', 'explained_variance', 'reconstruction_error'] + [col for col in results_df.columns if col not in ['n_components', 'explained_variance', 'reconstruction_error']]
    results_df = results_df[columns]

    return results_df
