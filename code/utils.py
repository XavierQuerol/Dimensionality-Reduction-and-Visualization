import sys

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import umap.umap_ as umap

### PREPROCESSING

def drop_columns(df, column_names):
    df = df.drop(columns = column_names)
    return df

def drop_rows_sick(df, column_names):
    df = df.dropna(subset=column_names)
    df = df[df["age"] <= 120] # Drop ages over 120
    df = df.reset_index(drop=True)

    return df

"""
Applies a minmaxscaler to all numerical columns.
If it finds a nan in a numerical column it removes the instance.
"""
def min_max_scaler(df, numerical_cols=slice(None)):

    scaler = MinMaxScaler()

    # Scaler Training with all the train and test information.
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def one_hot_encoding(df):
    categorical_features = df.select_dtypes(include=['object']).nunique()[lambda x: x > 2].index.tolist()

    ohe = OneHotEncoder(handle_unknown='ignore')

    encoded_array = ohe.fit_transform(df[categorical_features]).toarray()

    # Create new column names for the encoded features
    new_cols = [f'{col}_{cat}' for col in categorical_features for cat in
                ohe.categories_[categorical_features.index(col)]]

    # Create a DataFrame for the encoded features
    df_encoded = pd.DataFrame(encoded_array, columns=new_cols, index=df.index)

    # Substitute the original categorical features with the new numeric ones
    df = df.drop(categorical_features, axis=1)
    df = df.join(df_encoded)

    return df

def binary_encoding(df):

    binary_features = df.select_dtypes(include=['object']).nunique()[lambda x: x <= 2].index.tolist()

    # Encode only the binary features
    for feature in binary_features:

        label_encoder = LabelEncoder()
        df[feature] = label_encoder.fit_transform(df[feature])

    return df

def label_encoder(df):
    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])
    return df



def fill_nans(df, columns_predict):

    model = LinearRegression()

    # Train with all columns except the ones to predict
    cols = [col for col in df.columns if col not in columns_predict]

    for col in columns_predict:
        df_model = df.dropna(subset=[col])
        df_nans = df[df[col].isna()]

        if not df_model.empty:
            x = df_model[cols]
            y = df_model[col]

            model.fit(x, y)

            if not df_nans.empty:
                df.loc[df_nans.index, col] = model.predict(df_nans[cols])

    return df

#### UTILS REGARDING INPUT

def get_user_choice(prompt, options, is_numeric = False, is_float = False):
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            if is_numeric:
                print(f"  {option}")
            else:
                print(f" {i}. {option}")
        choice = input("Please enter the number of your choice: ")

        if is_numeric and is_float and float(choice) in options:
            return float(choice)
        if not is_float and is_numeric and int(choice) in options:
            return int(choice)
        if choice in options:
            return choice
        if not is_numeric and choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Invalid choice. Try again.\n")

def loading_bar(iteration, total, length=40):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '#' * bar_length + '-' * (length - bar_length)
    sys.stdout.write(f'\r[{bar}] {percent:.2%} Complete')
    sys.stdout.flush()

### PLOTS

def reduce_and_plot_with_umap(data, labels=None, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    """
    Reduces dimensionality of the dataset with UMAP and plots the data.
    Optionally, it plots clusters if labels are provided.

    :param data: Original high-dimensional data.
    :param labels: (Optional) Cluster labels corresponding to each data point.
    :param n_neighbors: UMAP parameter for balancing local and global structure.
    :param min_dist: UMAP parameter controlling the tightness of embedding.
    :param n_components: Number of dimensions to reduce to (default is 2 for visualization).
    :param metric: Distance metric for UMAP.
    """

    # Reduce dimensionality using UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    reduced_data = reducer.fit_transform(data)

    if n_components not in [2, 3]:
        print(f"UMAP with n_components={n_components} is not supported for visualization. Consider using n_components=2 or 3.")
        return

    fig = plt.figure(figsize=(10, 8) if n_components == 3 else (8, 6))

    if n_components == 2:
        ax = plt.gca()
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, alpha=0.6, s=30, cmap='viridis')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
    else:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, alpha=0.6, s=30, cmap='viridis')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_zlabel('UMAP Dimension 3')

    if labels is not None:
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)

    ax.set_title(f"Data Visualized in {'3D' if n_components == 3 else '2D'} Space (UMAP)")
    plt.grid(True) if n_components == 2 else None
    plt.show()


def reduce_and_plot_with_pca(data, labels=None, n_components=2):
    """
    Reduces dimensionality of the dataset with PCA and plots the data.
    Optionally, it plots clusters if labels are provided.

    :param data: Original high-dimensional data.
    :param labels: (Optional) Cluster labels corresponding to each data point.
    :param n_components: Number of dimensions to reduce to (default is 2 for visualization).
    """

    # Reduce dimensionality using PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)

    if n_components not in [2, 3]:
        print(f"PCA with n_components={n_components} is not supported for visualization. Consider using n_components=2 or 3.")
        return

    fig = plt.figure(figsize=(10, 8) if n_components == 3 else (8, 6))

    if n_components == 2:
        ax = plt.gca()
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, alpha=0.6, s=30, cmap='viridis')
        ax.set_xlabel('PCA Dimension 1')
        ax.set_ylabel('PCA Dimension 2')
    else:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, alpha=0.6, s=30, cmap='viridis')
        ax.set_xlabel('PCA Dimension 1')
        ax.set_ylabel('PCA Dimension 2')
        ax.set_zlabel('PCA Dimension 3')

    if labels is not None:
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)

    ax.set_title(f"Data Visualized in {'3D' if n_components == 3 else '2D'} Space (PCA)")
    plt.grid(True) if n_components == 2 else None
    plt.show()

