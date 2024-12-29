from utils import reduce_and_plot_with_pca, reduce_and_plot_with_umap
import matplotlib.pyplot as plt

from pca import customPCA
from global_fastkmeans import run_global_kmeans
from optics import apply_optics
from sklearn.decomposition import KernelPCA
import umap.umap_ as umap

import pandas as pd
import seaborn as sns
import numpy as np

from preprocessing import get_dataset


def plot_pairplots():
    for dataset, n_components in zip(['vowel', 'sick'], [21, 15]):
        # Load dataset
        X, y = get_dataset(dataset)

        # PCA reduction
        pca = customPCA()
        df_plot_pca = pca.reduce_dim(X, n_components=5)

        # Pairplot for PCA-reduced features
        sns.pairplot(pd.DataFrame(df_plot_pca[:, :5], columns=[f'PC{i+1}' for i in range(5)]))
        plt.title(f"Pairplot of PCA-Reduced Features for {dataset}")
        plt.show()

        # Compute variances and select top 5 features
        feature_variances = X.var(axis=0)
        top_5_indices = np.argsort(feature_variances)[-5:][::-1]  # Indices of 5 features with highest variance
        top_5_features = X.loc[:, top_5_indices.index]
        top_5_columns = [f'Feature_{i+1}' for i in top_5_indices]

        # Pairplot for original features with highest variance
        sns.pairplot(pd.DataFrame(top_5_features, columns=top_5_columns))
        plt.title(f"Pairplot of Top 5 High-Variance Features for {dataset}")
        plt.show()



def plot_reductions():

    for dataset, n_components in zip(['vowel', 'sick'], [21, 15]):
        df, y = get_dataset(dataset)

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

    hardcoded_results = {'means': {'sick': {"Davies-Bouldin Index": 1.09, "Silhouette Coefficient": 0.35, "Calinski": 843.32},
                                    'vowel': {"Davies-Bouldin Index": 1.17, "Silhouette Coefficient": 0.13 , "Calinski": 201.81}},
                         'optics': {'sick': {"Davies-Bouldin Index": 1.274, "Silhouette Coefficient": 0.523, "Calinski": 120.525},
                                    'vowel': {"Davies-Bouldin Index": 0.934, "Silhouette Coefficient": 0.108 , "Calinski": 271.772}}}

    metrics = ["Explained variance", "Davies-Bouldin Index", "Silhouette Coefficient", "Calinski"]
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharex=False)
    for red_method, marker, color_m in zip(['KERNEL', 'PCA'], ['o', '.'], ['red', 'orange']):
        
        for row_idx, dataset in enumerate(['vowel', 'sick']):
            for col_idx, metric in enumerate(metrics):  # Change col_idx to row_idx for metrics
                for color, cl_marker, cl_method in zip(['blue', 'green'], ['^', '*'], ['means', 'optics']):
                    df = pd.read_csv(f'./output/{red_method}_{dataset}_{cl_method}.csv')
                    df['Explained variance'] = df['explained_variance']
                    df = df.iloc[2:, :]
                    ax = axes[row_idx, col_idx] 
                    if metric == 'Explained variance':
                        if cl_method == 'optics':
                            continue
                        label = f'{red_method}'
                        cl_marker = marker
                    else:
                        label = f'{red_method} + {cl_method}'


                    if metric != 'Explained variance' and red_method=='KERNEL':
                        #ax.scatter(np.array(df['n_components'].max()), np.array(hardcoded_results[cl_method][dataset][metric]), color='black', marker=cl_marker, label=f'Original + {cl_method}', alpha=0.7)
                        if cl_method == 'means':
                            linestyle = '-'
                        else:
                            linestyle = 'dashed'
                        ax.hlines(y=np.array(hardcoded_results[cl_method][dataset][metric]), xmin=0, xmax=np.array(df['n_components'].max()), color='black', linestyle=linestyle, label=f'Original + {cl_method}', alpha=0.5)

                    
                    ax.scatter(df['n_components'], df[metric], color=color_m, marker=cl_marker, label=label, alpha=0.7)
                    ax.plot(df['n_components'], df[metric], color=color_m, alpha=0.7)

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

plot_metrics()