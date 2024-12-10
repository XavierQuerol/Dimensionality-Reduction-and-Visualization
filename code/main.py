import os
import pandas as pd
import matplotlib.pyplot as plt

from code.preprocessing import get_dataset
from code.pca import PCA
from utils import get_user_choice


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
    top_features = correlation_with_target.abs().sort_values(ascending=False).iloc[1:4]
    top_features = top_features.index.tolist()
    ds = ds.drop('target', axis=1)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(ds[top_features[0]], ds[top_features[1]], c=ds[top_features[2]], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label=f'{top_features[2]}')
    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])
    plt.title(f'Dataset {dataset} only representing the 3 components with more correlation with the target')
    plt.show()

    # Step 3 & 4 & 5 & 6 & 7 & 9
    pca = PCA(verbose=True)
    X_transformed = pca.reduce_dim(ds, threshold=0.90, n_components=3)

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


def  main():
    while True:
        option = get_user_choice("Which datset would you like to visualize?",["sick","vowel"])
        part_1(option)

        x = get_user_choice("Do you want to exit?", ["y", "n"])
        if x == "y":
            exit()

if __name__ == "__main__":
    main()