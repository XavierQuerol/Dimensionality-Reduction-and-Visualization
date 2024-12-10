import numpy as np

class PCA:
    def __init__(self):
        """
        Constructor

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.mean = None
        self.components = None  # Principal components
        self.remaining_components = None # Components remaining after PCA

        self.explained_variance = None  # Explained variance of each component in %
        self.eigenvalues = None # Eigenvalues of the principal components (amount of explained variance)

    def fit(self, X):
        """
        Fit the PCA model to the dataset X.

        Parameters:
        X (numpy.ndarray): Dataset with shape (n_samples, n_features).
        """
        # 1. Center dataset removing the mean
        self.mean = np.mean(X, axis=0)
        x_centered = X - self.mean

        # 2. Compute the covariance matrix
        cov_matrix = np.cov(x_centered, rowvar=False)

        # 3. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues / total_variance
        self.eigenvalues = eigenvalues
        self.components = eigenvectors
        self.remaining_components = eigenvectors

    def project(self, X):
        """
        Project the dataset X onto the principal components.

        Returns:
        numpy.ndarray: Transformed dataset with shape (n_samples, n_components).
        """
        x_centered = X - self.mean
        return np.dot(x_centered, self.remaining_components)

    def reduce_dim(self, X, variability_threshold=0.95):
        """
        Fits the dataset and returns its projected version.

        Parameters:
        X (numpy.ndarray): Dataset with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Transformed dataset with shape (n_samples, n_components).
        """
        self.fit(X)

        cumulative_variance_ratio = np.cumsum(self.explained_variance)
        n_components = np.searchsorted(cumulative_variance_ratio, variability_threshold) + 1

        self.remaining_components = self.components[:, :n_components]

        return self.project(X)
