from utils import reduce_and_plot_with_pca, reduce_and_plot_with_umap

import pandas as pd
import seaborn as sns

df = pd.read_csv('./datasets_processed/sick.csv')

#sns.pairplot(df[['age', 'TSH', 'T3', 'TT4']])
reduce_and_plot_with_pca(df[:-1], n_components=2)
reduce_and_plot_with_pca(df[:-1], n_components=3)
reduce_and_plot_with_umap(df[:-1], n_components=2)
reduce_and_plot_with_umap(df[:-1], n_components=3)
