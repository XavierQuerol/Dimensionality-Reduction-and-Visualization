import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

def pca_with_threshold(dataset, threshold=0.95):

    pca = PCA(n_components=threshold)  # Threshold will automatically select the number of components
    pca.fit(dataset)

    # Get the number of components that explain the desired threshold of variance
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold) + 1

    # Perform the PCA with the calculated number of components
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(dataset)

    return pca, transformed_data


def incremental_pca_with_threshold(dataset, threshold=0.95, batch_size=10):

    # Perform IncrementalPCA
    ipca = IncrementalPCA(n_components=dataset.shape[1], batch_size=batch_size)
    ipca.fit(dataset)

    # Calculate the cumulative variance and find the required number of components
    cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    # Perform IncrementalPCA with the number of components required
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    transformed_data = ipca.fit_transform(dataset)

    return ipca, transformed_data

def kernel_pca_with_threshold(dataset, threshold=0.95, kernel='rbf', gamma=None):

    # Perform KernelPCA with the specified kernel
    kpca = KernelPCA(kernel=kernel, gamma=gamma, n_components=dataset.shape[1])
    kpca.fit(dataset)

    # Calculate the cumulative explained variance ratio and find the required number of components
    cumulative_variance = np.cumsum(kpca.lambdas_ / np.sum(kpca.lambdas_))  # Normalized eigenvalues as variance explained
    n_components = np.argmax(cumulative_variance >= threshold) + 1  # +1 to adjust for 0-indexing

    # Perform KernelPCA with the number of components required
    kpca = KernelPCA(kernel=kernel, gamma=gamma, n_components=n_components)
    transformed_data = kpca.fit_transform(dataset)

    return kpca, transformed_data
