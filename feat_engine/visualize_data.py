import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from typing import Union


class DataVisualizer:
    """
    A class for visualizing different aspects of the dataset, including distribution of features,
    feature interactions, outlier detection, temporal data, and dimensionality reduction.
    """

    def __init__(self) -> None:
        pass

    # 1. General Data Exploration
    def plot_distribution(self, df: pd.DataFrame, columns: list, kind: str = 'histogram') -> None:
        """
        Plot the distribution of specified columns in the dataframe.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - columns (list): List of column names to plot.
        - kind (str): Type of plot ('histogram', 'kde', or 'box').
        """
        for col in columns:
            plt.figure(figsize=(8, 4))
            if kind == 'histogram':
                sns.histplot(df[col], kde=True)
            elif kind == 'kde':
                sns.kdeplot(df[col], shade=True)
            elif kind == 'box':
                sns.boxplot(df[col])
            else:
                raise ValueError(f"Unsupported kind: {kind}")
            plt.title(f'Distribution of {col}')
            plt.show()

    def plot_missing_data(self, df: pd.DataFrame) -> None:
        """
        Visualize missing data in the dataframe using a heatmap.

        Args:
        - df (pd.DataFrame): Input dataframe.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Data Heatmap")
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Plot a heatmap of correlations between numerical features in the dataframe.

        Args:
        - df (pd.DataFrame): Input dataframe.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    # 2. Feature Interactions
    def plot_pairwise_relationships(self, df: pd.DataFrame, columns: list) -> None:
        """
        Plot pairwise relationships between features.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - columns (list): List of column names to plot pairwise relationships.
        """
        sns.pairplot(df[columns], diag_kind="kde")
        plt.show()

    def plot_scatter_with_outliers(self, df: pd.DataFrame, x: str, y: str, outliers: pd.Series) -> None:
        """
        Plot scatter plot with outliers highlighted.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - x (str): Name of the x-axis feature.
        - y (str): Name of the y-axis feature.
        - outliers (pd.Series): Boolean series indicating outliers.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x], df[y], c=outliers, cmap='coolwarm', edgecolor='k')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Scatter plot of {x} vs {y} with Outliers')
        plt.show()

    # 3. Outlier Detection Visualization
    def plot_boxplot_with_outliers(self, df: pd.DataFrame, columns: list) -> None:
        """
        Plot boxplots for columns to visualize potential outliers.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - columns (list): List of column names to plot.
        """
        plt.figure(figsize=(12, 6))
        df[columns].boxplot()
        plt.title("Box Plot for Outlier Detection")
        plt.show()

    def plot_isolation_forest_outliers(self, df: pd.DataFrame, outliers: pd.Series) -> None:
        """
        Highlight outliers detected by Isolation Forest in a scatter plot.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - outliers (pd.Series): Boolean series indicating outliers.
        """
        px.scatter(df, x=df.columns[0], y=df.columns[1], color=outliers, title='Isolation Forest Outliers').show()

    # 4. Temporal Data Visualization
    def plot_time_series(self, df: pd.DataFrame, date_col: str, value_col: str, rolling_window: Union[int, None] = None) -> None:
        """
        Plot time series data with an optional rolling window.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - date_col (str): Name of the datetime column.
        - value_col (str): Name of the value column to plot.
        - rolling_window (int): Optional rolling window size.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(df[date_col], df[value_col], label='Original Data')
        if rolling_window:
            plt.plot(df[date_col], df[value_col].rolling(window=rolling_window).mean(), label=f'Rolling Mean ({rolling_window})')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.title(f'Time Series of {value_col}')
        plt.legend()
        plt.show()

    # 5. Dimensionality Reduction Visualization
    def plot_pca(self, df: pd.DataFrame, n_components: int = 2) -> None:
        """
        Plot the results of Principal Component Analysis (PCA).

        Args:
        - df (pd.DataFrame): Input dataframe.
        - n_components (int): Number of components to reduce to.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_df)
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c='b')
        plt.title('PCA Result')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

    def plot_tsne(self, df: pd.DataFrame, n_components: int = 2, perplexity: int = 30) -> None:
        """
        Plot the results of t-SNE dimensionality reduction.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - n_components (int): Number of components to reduce to.
        - perplexity (int): Perplexity parameter for t-SNE.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        tsne_result = tsne.fit_transform(numeric_df)
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='r')
        plt.title('t-SNE Result')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

    def plot_umap(self, df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> None:
        """
        Plot the results of UMAP dimensionality reduction.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - n_components (int): Number of components to reduce to.
        - n_neighbors (int): The size of the local neighborhood (in terms of number of neighboring points).
        - min_dist (float): Minimum distance between points in the low-dimensional space.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # UMAP model
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_result = umap_model.fit_transform(numeric_df)

        # Plot UMAP result
        plt.figure(figsize=(8, 6))
        plt.scatter(umap_result[:, 0], umap_result[:, 1], c='g')
        plt.title('UMAP Result')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.show()

    # 6. Interactive Visualizations using Plotly
    def plot_interactive_histogram(self, df: pd.DataFrame, column: str) -> None:
        """
        Create an interactive histogram using Plotly.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - column (str): Column to visualize.
        """
        fig = px.histogram(df, x=column)
        fig.show()

    def plot_interactive_correlation(self, df: pd.DataFrame) -> None:
        """
        Create an interactive correlation heatmap using Plotly.

        Args:
        - df (pd.DataFrame): Input dataframe.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.index,
                                        colorscale='Viridis'))
        fig.update_layout(title="Interactive Correlation Heatmap")
        fig.show()

    # 7. Interactive Scatter Plots
    def plot_interactive_scatter(self, df: pd.DataFrame, x: str, y: str, color: Union[str, None] = None, size: Union[str, None] = None) -> None:
        """
        Create an interactive scatter plot using Plotly.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - x (str): X-axis column.
        - y (str): Y-axis column.
        - color (str): Column for color encoding.
        - size (str): Column for size encoding.
        """
        fig = px.scatter(df, x=x, y=y, color=color, size=size)
        fig.show()

    # 8. Feature Importance Visualization
    def plot_feature_importance(self, feature_importances: np.ndarray, feature_names: list) -> None:
        """
        Plot feature importance from a machine learning model.

        Args:
        - feature_importances (np.ndarray): Array of feature importance values.
        - feature_names (list): List of feature names.
        """
        indices = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(feature_importances)), feature_importances[indices], align='center')
        plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    # 9. Categorical Data Visualization
    def plot_categorical_distribution(self, df: pd.DataFrame, column: str, hue: Union[str, None] = None) -> None:
        """
        Plot the distribution of a categorical feature.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - column (str): Name of the categorical column.
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(df, x=column, hue=hue)
        plt.title(f'Distribution of {column}')
        plt.show()

    # 10. Plot Target Distribution
    def plot_target_distribution(self, df: pd.DataFrame, target_column: str) -> None:
        """
        Plot the distribution of a target variable (for classification or regression tasks).

        Args:
        - df (pd.DataFrame): Input dataframe.
        - target_column (str): Name of the target column.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target_column], kde=True)
        plt.title(f'Target Distribution: {target_column}')
        plt.show()
