from utils import reduce_and_plot_with_pca, reduce_and_plot_with_umap
import matplotlib.pyplot as plt

from pca import customPCA
from global_fastkmeans import run_global_kmeans
from optics import apply_optics
from sklearn.decomposition import KernelPCA
import umap.umap_ as umap

import pandas as pd
import seaborn as sns

for dataset, n_components in zip(['vowel', 'sick'], [21, 15]    ):
    df = pd.read_csv(f'./datasets_processed/{dataset}.csv')

    pca = customPCA()
    kpca = KernelPCA(kernel='rbf', n_components=n_components, fit_inverse_transform=True)
    reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')

    df_plot = pca.reduce_dim(df, n_components=2)
    df_plot_umap = reducer_umap.fit_transform(df)

    df_reduced_pca = pca.reduce_dim(df, n_components=n_components)
    df_reduced_kpca = kpca.fit_transform(df)
    

    _, labels_gkmeans = run_global_kmeans(df, max_clusters=12, distance= 'euclidean')
    _, labels_gkmeans_reduced = run_global_kmeans(df_reduced_pca, max_clusters=12, distance= 'euclidean')

    optics_labels = apply_optics(df, metric='minkowski', algorithm='ball_tree')
    optics_labels_reduced = apply_optics(df_reduced_pca, metric='minkowski', algorithm='ball_tree')

    fig, axes = plt.subplots(2, 4, figsize=(12, 8))

    axes[0, 0].scatter(df_plot[:, 0], df_plot[:, 1], c=labels_gkmeans, alpha=0.6, s=30, cmap='tab20')
    axes[0, 1].scatter(df_plot[:, 0], df_plot[:, 1], c=labels_gkmeans_reduced, alpha=0.6, s=30, cmap='tab20')
    axes[0, 2].scatter(df_plot[:, 0], df_plot[:, 1], c=optics_labels, alpha=0.6, s=30, cmap='tab20')
    axes[0, 3].scatter(df_plot[:, 0], df_plot[:, 1], c=optics_labels_reduced, alpha=0.6, s=30, cmap='tab20')
    axes[0, 0].set_title(f"Global Kmeans", fontsize=12)
    axes[0, 1].set_title(f"Global Kmeans reduced", fontsize=12)
    axes[0, 2].set_title(f"OPTICS", fontsize=12)
    axes[0, 3].set_title(f"OPTICS reduced", fontsize=12)
    axes[0, 0].set_ylabel(f"Using PCA", fontsize=12)
    axes[1, 0].set_ylabel(f"Using UMAP", fontsize=12)

    axes[1, 0].scatter(df_plot_umap[:, 0], df_plot_umap[:, 1], c=labels_gkmeans, alpha=0.6, s=30, cmap='tab20')
    axes[1, 1].scatter(df_plot_umap[:, 0], df_plot_umap[:, 1], c=labels_gkmeans_reduced, alpha=0.6, s=30, cmap='tab20')
    axes[1, 2].scatter(df_plot_umap[:, 0], df_plot_umap[:, 1], c=optics_labels, alpha=0.6, s=30, cmap='tab20')
    axes[1, 3].scatter(df_plot_umap[:, 0], df_plot_umap[:, 1], c=optics_labels_reduced, alpha=0.6, s=30, cmap='tab20')

    plt.tight_layout()
    plt.show()






#sns.pairplot(df[['age', 'TSH', 'T3', 'TT4']])
"""reduce_and_plot_with_pca(df[:-1], n_components=2)
reduce_and_plot_with_pca(df[:-1], n_components=3)
reduce_and_plot_with_umap(df[:-1], n_components=2)
reduce_and_plot_with_umap(df[:-1], n_components=3)"""

def plot_metrics():
    metrics = ["Explained variance", "Davies-Bouldin Index", "Silhouette Coefficient", "Calinski"]
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharex=True)
    for red_method, marker, color_m in zip(['KERNEL', 'PCA'], ['^', '*'], ['black', 'grey']):
        
        for row_idx, dataset in enumerate(['vowel', 'sick']):
            for col_idx, metric in enumerate(metrics):  # Change col_idx to row_idx for metrics
                for color, cl_method in zip(['red', 'orange'], ['means', 'optics']):
                    df = pd.read_csv(f'./output/{red_method}_{dataset}_{cl_method}.csv')
                    df['Explained variance'] = df['explained_variance']
                    df = df.iloc[2:, :]
                    ax = axes[row_idx, col_idx] 
                    if metric == 'Explained variance':
                        if cl_method == 'optics':
                            continue
                        color = color_m
                        label = f'{red_method}'
                    else:
                        label = f'{red_method} + {cl_method}'

                    ax.scatter(df['n_components'], df[metric], color=color, marker=marker, label=label, alpha=0.7)
                    ax.plot(df['n_components'], df[metric], color=color, alpha=0.7)

                # Format the axis values to one decimal place for Davies-Bouldin and Silhouette Coefficient
                if metric in ["Davies-Bouldin Index", "Silhouette Coefficient"]:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))  # One decimal for y-axis

                # No decimals for cluster numbers (k) on the x-axis
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

                # Display all k values on x-axis
                comp = sorted(df['n_components'].unique())
                comp = [r for r in range(min(comp), max(comp), 4)]
                ax.set_xticks(comp)  # Show all k values
                ax.tick_params(axis='x', labelsize=10)  # Set x-axis label size smaller

                # Add grid
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                ax.grid(axis="x", linestyle="--", alpha=0.5)

                # Set titles, axis names, and labels
                if col_idx == 0:
                    ax.set_ylabel(f"Dataset {dataset}", fontsize=12)  # Dataset name in the first row
                if row_idx == 0:
                    if metric == 'Davies-Bouldin Index':
                        ax.set_title(f"DBI", fontsize=12)  # Metric name for the Y-axis
                    else:
                        ax.set_title(f"{metric}", fontsize=12)  # Metric name for the Y-axis
                if row_idx == 1:
                    ax.set_xlabel("# Components", fontsize=12)  # X-axis label for the last row

                # Show legend only on the first row and last column
                if row_idx == 0 and col_idx == 0:
                    ax.legend(title="Method", fontsize=10)
                if row_idx == 0 and col_idx == 3:
                    ax.legend(title="Method", fontsize=10)

    # Adjust layout to prevent overlap and ensure proper spacing
    plt.tight_layout()
    plt.show()

#plot_metrics()