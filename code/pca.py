import numpy as np

class customPCA:
    def __init__(self, verbose=False):
        """
        Constructor

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.mean = None
        self.components = None  # Principal components
        self.remaining_components = None # Components remaining after PCA
        self.covariance_matrix = None # Covariance matrix

        self.explained_variance = None  # Explained variance of each component in %
        self.eigenvalues = None # Eigenvalues of the principal components (amount of explained variance)
        self.verbose = verbose

    def display_eigenvectors(self):
        self.compute_explained_variance()
        print("Principal Components:")
        for i, c in enumerate(self.components.T):
            eigenvectors = ", ".join([f"{component:.4f}" for component in c])
            print(
                f"Eigenvector ({i}): {eigenvectors} - Eigenvalue: {self.eigenvalues[i]:.4f} - Represents {self.explained_variance[i] * 100:.4f}% of variance.")

    def compute_explained_variance(self):
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance = self.eigenvalues / total_variance

    def fit(self, X):
        """
        Fit the PCA model to the dataset X.

        Parameters:
        X (numpy.ndarray): Dataset with shape (n_samples, n_features).
        """
        # 1. Center dataset removing the mean
        if self.verbose:
            print("STEP 3: DONE")
        self.mean = np.mean(X, axis=0)
        x_centered = X - self.mean

        # 2. Compute the covariance matrix
        self.covariance_matrix = np.cov(x_centered, rowvar=False)

        if self.verbose:
            print("STEP 4: WRITE COVARIANCE MATRIX")
            print(self.covariance_matrix)

        # 3. Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)

        self.eigenvalues = eigenvalues
        self.components = eigenvectors

        if self.verbose: # STEP 5
            print("STEP 5: WRITE EIGENVECTORS")
            self.display_eigenvectors()

        # 4. Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.eigenvalues = eigenvalues
        self.components = eigenvectors
        self.remaining_components = eigenvectors
        self.compute_explained_variance()

        if self.verbose: # STEP 6
            print("STEP 6: WRITE SORTED EIGENVECTORS")
            self.display_eigenvectors()

    def project(self, X):
        """
        Project the dataset X onto the principal components.

        Returns:
        numpy.ndarray: Transformed dataset with shape (n_samples, n_components).
        """
        x_centered = X - self.mean
        return np.dot(x_centered, self.remaining_components)

    def reduce_dim(self, X, threshold=0.95, n_components=None):
        """
        Fits the dataset and returns its projected version.

        Parameters:
        X (numpy.ndarray): Dataset with shape (n_samples, n_features).
        threshold (float): Threshold for determining which components are variability. If n_components is defined, it will be used. Otherwise it will use variability_threshold.
        n_components (int): Number of principal components to keep.

        Returns:
        numpy.ndarray: Transformed dataset with shape (n_samples, n_components).
        """
        self.fit(X)

        if n_components is None:
            cumulative_variance_ratio = np.cumsum(self.explained_variance)
            n_components = np.searchsorted(cumulative_variance_ratio, threshold) + 1

        self.remaining_components = self.components[:, :n_components]

        return self.project(X)

    def reconstruct(self, x_projected):
        """
        Reconstruct the original dataset from the projected data.

        Parameters:
        X_projected (numpy.ndarray): The data projected onto the principal components (shape: [n_samples, n_components]).

        Returns:
        numpy.ndarray: Reconstructed dataset with shape (n_samples, n_features).
        """
        # Multiply the projected data by the principal components
        X_reconstructed = np.dot(x_projected, self.remaining_components.T)

        # Add back the mean
        X_reconstructed += self.mean.values

        return X_reconstructed