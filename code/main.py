import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.cluster import OPTICS
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition

from code.preprocessing import get_dataset
from code.pca import PCA
from utils import get_user_choice
from global_kmeans import run_global_kmeans



def load_ds(name):

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_processed")
    file = os.path.join(base_dir,f'{name}.csv')
    df = pd.read_csv(file)

    # We remove the class of the dataset as we will not be using it
    return df.iloc[:,:-1], df.iloc[:,-1]

def part_1(dataset):
    # Step 1
    x, y = get_dataset(dataset)
    ds = x.copy()
    ds["target"] = y

    # Step 2
    correlation_with_target = ds.corr()["target"]  # Based on their correlation with the target
    top_features = correlation_with_target.abs().sort_values(ascending=False).iloc[1:4].index.tolist()

    ds = ds.drop('target', axis=1)
    feature_indices = [ds.columns.get_loc(feature) for feature in top_features]  # Get indices of the features

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        ds.iloc[:, feature_indices[0]],  # Access by column index
        ds.iloc[:, feature_indices[1]],  # Access by column index
        c=ds.iloc[:, feature_indices[2]],  # Access by column index
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Dataset {dataset} only representing the 3 components with more correlation with the target')
    plt.show()

    # Step 3 & 4 & 5 & 6 & 7
    pca = PCA(verbose=True)
    X_transformed = pca.reduce_dim(ds, n_components=3)

    # Step 8
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=X_transformed[:, 2], cmap='viridis',alpha=0.7)  # `viridis` is a color map, change as needed

    # Add a color bar to show the gradient
    plt.colorbar(scatter, label='Third Dimension (Color Gradient)')

    plt.xlabel('First Component (X)')
    plt.ylabel('Second Component (Y)')
    plt.title(f'Dataset {dataset} after applying PCA and leaving 3 principal components')
    plt.show()

    # Step 9
    X_reconstructed = pca.reconstruct(X_transformed)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_reconstructed[:, feature_indices[0]],  # Access by column index
        X_reconstructed[:, feature_indices[1]],  # Access by column index
        c=X_reconstructed[:, feature_indices[2]],  # Access by column index
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Dataset {dataset} only representing the 3 components with more correlation with the target')
    plt.show()

def sklearn_PCA(dataset):
    from sklearn import decomposition
    # Step 1
    x, y = get_dataset(dataset)
    ds = x.copy()
    ds["target"] = y

    # Step 2
    correlation_with_target = ds.corr()["target"]  # Based on their correlation with the target
    top_features = correlation_with_target.abs().sort_values(ascending=False).iloc[1:4].index.tolist()

    ds = ds.drop('target', axis=1)
    feature_indices = [ds.columns.get_loc(feature) for feature in top_features]  # Get indices of the features

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        ds.iloc[:, feature_indices[0]],  # Access by column index
        ds.iloc[:, feature_indices[1]],  # Access by column index
        c=ds.iloc[:, feature_indices[2]],  # Access by column index
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Dataset {dataset} only representing the 3 components with more correlation with the target')
    plt.show()

    # Step 3 & 4 & 5 & 6 & 7
    pca = decomposition.PCA(n_components=3)  # Reduce to 3 components
    X_transformed = pca.fit_transform(ds)

    # Step 8
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=X_transformed[:, 2], cmap='viridis',
                          alpha=0.7)  # `viridis` is a color map, change as needed

    # Add a color bar to show the gradient
    plt.colorbar(scatter, label='Third Dimension (Color Gradient)')

    plt.xlabel('First Component (X)')
    plt.ylabel('Second Component (Y)')
    plt.title(f'Dataset {dataset} after applying PCA and leaving 3 principal components')
    plt.show()

    # Step 9
    X_reconstructed = pca.inverse_transform(X_transformed)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_reconstructed[:, feature_indices[0]],  # Access by column index
        X_reconstructed[:, feature_indices[1]],  # Access by column index
        c=X_reconstructed[:, feature_indices[2]],  # Access by column index
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Dataset {dataset} only representing the 3 components with more correlation with the target')
    plt.show()

def sklearn_incremental_PCA(dataset):
    x, y = get_dataset(dataset)
    ds = x.copy()
    ds["target"] = y

    # Step 2: Select top features based on correlation with the target
    correlation_with_target = ds.corr()["target"]  # Correlation with target
    top_features = correlation_with_target.abs().sort_values(ascending=False).iloc[1:4].index.tolist()

    # Extract feature indices for later use
    ds = ds.drop('target', axis=1)
    feature_indices = [ds.columns.get_loc(feature) for feature in top_features]

    # Step 3: Apply Incremental PCA for dimensionality reduction
    # Define mini-batches
    batch_size = 100  # Set a batch size suitable for your system's memory
    incremental_pca = decomposition.IncrementalPCA(n_components=3)

    # Perform partial fit on mini-batches
    for i in range(0, ds.shape[0], batch_size):
        batch = ds.iloc[i:i + batch_size]
        incremental_pca.partial_fit(batch)

    # Transform the dataset in batches
    X_transformed = np.vstack([
        incremental_pca.transform(ds.iloc[i:i + batch_size])
        for i in range(0, ds.shape[0], batch_size)
    ])

    # Step 4: Scatter plot of PCA-transformed components
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_transformed[:, 0],  # First principal component
        X_transformed[:, 1],  # Second principal component
        c=X_transformed[:, 2],  # Third principal component (as color gradient)
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Third Principal Component (Color Gradient)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'Dataset {dataset} after Incremental PCA (3 Principal Components)')
    plt.show()

    # Step 5: Reconstruct the original dataset from Incremental PCA components
    X_reconstructed = np.vstack([
        incremental_pca.inverse_transform(X_transformed[i:i + batch_size])
        for i in range(0, X_transformed.shape[0], batch_size)
    ])

    # Step 6: Visualize reconstructed features
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_reconstructed[:, feature_indices[0]],  # Reconstructed first feature
        X_reconstructed[:, feature_indices[1]],  # Reconstructed second feature
        c=X_reconstructed[:, feature_indices[2]],  # Reconstructed third feature (as color gradient)
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'Reconstructed {top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Reconstructed Dataset {dataset} using Incremental PCA')
    plt.show()

def kernel_pca_clustering(dataset, kernel='linear', n_components=2, max_clusters_gkmeans=12, min_samples_optics=15):
      """
      Performs Kernel PCA, Global K-means, and OPTICS clustering.

      Args:
        dataset: The input dataset.
        kernel: The kernel to use for Kernel PCA (default: 'linear').
        n_components: The number of components to keep in Kernel PCA (default: 3).
        max_clusters_gkmeans: Maximum number of clusters for Global K-means.
        min_samples_optics: Minimum samples for OPTICS clustering.

      Returns:
        A tuple containing:
          - The dataset transformed by Kernel PCA.
          - Cluster labels from Global K-means.
          - Cluster labels from OPTICS.
      """

      # 1. Kernel PCA
      kpca = KernelPCA(kernel=kernel, n_components=n_components)
      dataset_kpca = kpca.fit_transform(dataset)
      scaler = MinMaxScaler()
      dataset_kpca_normalized = scaler.fit_transform(dataset_kpca)

      # 2. Global K-means
      clusters_gkmeans, labels_gkmeans = run_global_kmeans(dataset_kpca_normalized, max_clusters=max_clusters_gkmeans)

      # 3. OPTICS
      optics = OPTICS(min_samples=min_samples_optics)
      labels_optics = optics.fit_predict(dataset_kpca_normalized)

      return dataset_kpca, labels_gkmeans, labels_optics

def  main():
    while True:
        dataset = get_user_choice("Which datset would you like to visualize?",["sick","vowel"])
        algorithm = get_user_choice("Which algorithm would you like to use?",["custom PCA","sklearn.decomposition.PCA","sklearn.decomposition.IncrementalPCA"])

        if algorithm == "custom PCA":
            part_1(dataset)
        elif algorithm == "sklearn.decomposition.PCA":
            sklearn_PCA(dataset)
        elif algorithm == "sklearn.decomposition.IncrementalPCA":
            sklearn_incremental_PCA(dataset)

        x = get_user_choice("Do you want to exit?", ["y", "n"])
        if x == "y":
            exit()

if __name__ == "__main__":
    main()