import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from code.preprocessing import get_dataset
from code.pca_analysis import pca_analysis, customPCA_analysis, incremental_pca_analysis
from code.pca import customPCA
from code.global_kmeans import run_global_kmeans
from code.utils import get_user_choice

from sklearn import decomposition
from tqdm import tqdm

from sklearn.decomposition import KernelPCA
from sklearn.cluster import OPTICS
from sklearn.preprocessing import MinMaxScaler


def load_ds(name):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets_processed")
    file = os.path.join(base_dir,f'{name}.csv')
    df = pd.read_csv(file)
    # We remove the class of the dataset as we will not be using it
    return df.iloc[:,:-1], df.iloc[:,-1]

def create_pca_matrix(ds):
    results = []

    for comp in tqdm(range(1, ds.shape[1]),desc="Calculating explained variance"):
        tab = compare_pca_models(ds, n_components=comp, batch_size=ds.shape[1])
        new_row = [
            comp,
            np.sum(tab.iloc[0]["Explained Variance"]), tab.iloc[0]["Reconstruction Error"],
            np.sum(tab.iloc[1]["Explained Variance"]), tab.iloc[1]["Reconstruction Error"],
            np.sum(tab.iloc[2]["Explained Variance"]), tab.iloc[2]["Reconstruction Error"],
        ]

        results.append(new_row)

    columns = [
        "Components",
        "cPCA Variability", "cPCA Reconstruction Error",
        "PCA Variability", "PCA Reconstruction Error",
        "iPCA Variability", "iPCA Reconstruction Error"
    ]

    # Create a DataFrame from the data
    return pd.DataFrame(results, columns=columns)

def plot_models(dataset):
    x, _ = get_dataset(dataset)
    ds = x.copy()
    df = create_pca_matrix(ds)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['Components'], df['cPCA Variability'], label='cPCA Variability', color='blue', linestyle='-')
    ax1.plot(df['Components'], df['PCA Variability'], label='PCA Variability', color='green', linestyle='-')
    ax1.plot(df['Components'], df['iPCA Variability'], label='iPCA Variability', color='red', linestyle='-')

    ax1.set_xlabel('Components')
    ax1.set_ylabel('cPCA Variability, PCA Variability, iPCA Variability', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()

    ax2.plot(df['Components'], df['cPCA Reconstruction Error'], label='cPCA Reconstruction Error', color='orange',
             linestyle='--')
    ax2.plot(df['Components'], df['PCA Reconstruction Error'], label='PCA Reconstruction Error', color='purple',
             linestyle='--')
    ax2.plot(df['Components'], df['iPCA Reconstruction Error'], label='iPCA Reconstruction Error', color='brown',
             linestyle='--')

    ax2.set_ylabel('cPCA Reconstruction Error, PCA Reconstruction Error, iPCA Reconstruction Error', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Explainability of each PCA method')
    plt.grid(True)

    plt.show()


def compare_pca_models(X, n_components=3, batch_size=10):
    custompca_results = customPCA_analysis(X, n_components=n_components)
    pca_results = pca_analysis(X, n_components=n_components)
    ipca_results = incremental_pca_analysis(X, n_components=n_components, batch_size=batch_size)

    # Concatenate the results into one DataFrame
    comparison_df = pd.concat([custompca_results, pca_results, ipca_results], ignore_index=True)

    return comparison_df

def part_1(dataset):
    # Step 1
    print("STEP 1: DONE")
    x, _ = get_dataset(dataset)
    ds = x.copy()
    # Step 2
    variances = ds.apply(lambda col: np.var(col, ddof=1))  # ddof=1 for sample variance instead of population variance
    top_features = variances.sort_values(ascending=False).iloc[:3].index.tolist()
    feature_indices = [ds.columns.get_loc(feature) for feature in top_features]  # Get indices of the features

    total_variance = variances.sum()
    print(f"Total Variance: {total_variance:.2f}")
    for feature in top_features:
        explained_variance = variances[feature]
        percentage_explained = (explained_variance / total_variance) * 100
        print(f"Feature: {feature} - Variance: {explained_variance:.2f} - "
              f"Percentage of Total Variance Explained: {percentage_explained:.2f}%")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        ds.iloc[:, feature_indices[0]],
        ds.iloc[:, feature_indices[1]],
        c=ds.iloc[:, feature_indices[2]],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'STEP 2: Dataset {dataset} only representing 3 features with more variance')
    plt.show()

    # Step 3 & 4 & 5 & 6 & 7
    pca = customPCA(verbose=True)
    X_transformed = pca.reduce_dim(ds, n_components=3)
    print("STEP 7: DONE")

    # Step 8
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=X_transformed[:, 2], cmap='viridis',alpha=0.7)  # `viridis` is a color map, change as needed
    plt.colorbar(scatter, label='Third Dimension (Color Gradient)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'STEP 8: Dataset {dataset} after applying PCA and leaving 3 principal components')
    plt.show()
    # Step 9
    X_reconstructed = pca.reconstruct(X_transformed)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_reconstructed[:, feature_indices[0]],
        X_reconstructed[:, feature_indices[1]],
        c=X_reconstructed[:, feature_indices[2]],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'STEP 9: Dataset {dataset} reconstructed representing the 3 components with more variance')
    plt.show()
def sklearn_PCA(dataset):
    # Step 1
    x, _ = get_dataset(dataset)
    ds = x.copy()
    variances = ds.apply(lambda col: np.var(col, ddof=1))  # ddof=1 for sample variance instead of population variance
    top_features = variances.sort_values(ascending=False).iloc[:3].index.tolist()
    feature_indices = [ds.columns.get_loc(feature) for feature in top_features]  # Get indices of the features
    # Step 3 & 4 & 5 & 6 & 7
    pca = decomposition.PCA(n_components=3)  # Reduce to 3 components
    X_transformed = pca.fit_transform(ds)
    # Step 8
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=X_transformed[:, 2], cmap='viridis',
                          alpha=0.7)
    # Add a color bar to show the gradient
    plt.colorbar(scatter, label='Third Dimension (Color Gradient)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'Dataset {dataset} after applying PCA and leaving 3 principal components')
    plt.show()
    # Step 9
    X_reconstructed = pca.inverse_transform(X_transformed)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_reconstructed[:, feature_indices[0]],
        X_reconstructed[:, feature_indices[1]],
        c=X_reconstructed[:, feature_indices[2]],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Dataset {dataset} reconstructed representing the 3 components with more variance')
    plt.show()
def sklearn_incremental_PCA(dataset):
    x, _ = get_dataset(dataset)
    ds = x.copy()
    variances = ds.apply(lambda col: np.var(col, ddof=1))  # ddof=1 for sample variance instead of population variance
    top_features = variances.sort_values(ascending=False).iloc[:3].index.tolist()
    feature_indices = [ds.columns.get_loc(feature) for feature in top_features]
    # Incremental PCA
    # Define mini-batches
    batch_size = 100
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
        X_transformed[:, 0],
        X_transformed[:, 1],
        c=X_transformed[:, 2],
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
        X_reconstructed[:, feature_indices[0]],
        X_reconstructed[:, feature_indices[1]],
        c=X_reconstructed[:, feature_indices[2]],
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
        n_components: The number of components to keep in Kernel PCA (default: 2).
        max_clusters_gkmeans: Maximum number of clusters for Global K-means.
        min_samples_optics: Minimum samples for OPTICS clustering.
        xi: Determines the minimum steepness on the reachability plot for OPTICS clustering.
        min_cluster_size: Minimum cluster size for OPTICS clustering.


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
        function = get_user_choice("What do you want to do?", ["Compare PCA models explainability", "Visualize individual datasets using PCA"])
        dataset = get_user_choice("Which datset would you like to visualize?",["sick","vowel"])

        if function == "Visualize individual datasets using PCA":
            algorithm = get_user_choice("Which algorithm would you like to use?",["custom PCA","sklearn.decomposition.PCA","sklearn.decomposition.IncrementalPCA"])
            if algorithm == "custom PCA":
                part_1(dataset)
            elif algorithm == "sklearn.decomposition.PCA":
                sklearn_PCA(dataset)
            elif algorithm == "sklearn.decomposition.IncrementalPCA":
                sklearn_incremental_PCA(dataset)
        elif function == "Compare PCA models explainability":
            plot_models(dataset)
        x = get_user_choice("Do you want to exit?", ["y", "n"])
        if x == "y":
            exit()
if __name__ == "__main__":
    main()