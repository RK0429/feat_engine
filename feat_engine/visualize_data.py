import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from typing import List, Optional


class DataVisualizer:
    """
    A class for visualizing different aspects of the dataset, including distributions, feature interactions,
    outlier detection, temporal data, dimensionality reduction, and more.

    Methods:
    - plot_distribution: Plot the distribution of specified columns.
    - plot_missing_data: Visualize missing data in the dataframe.
    - plot_correlation_heatmap: Plot a heatmap of correlations between numerical features.
    - plot_swarmplot: Create a swarmplot to visualize data distribution across categories.
    - plot_3d_scatter: Create a 3D scatter plot for three numerical features.
    - plot_pairwise_relationships: Plot pairwise relationships between features.
    - plot_scatter_with_outliers: Plot scatter plot with outliers highlighted.
    - plot_boxplot_with_outliers: Plot boxplots for columns to visualize potential outliers.
    - plot_isolation_forest_outliers: Highlight outliers detected by Isolation Forest.
    - plot_time_series: Plot time series data with optional rolling window.
    - plot_pca: Plot the results of Principal Component Analysis.
    - plot_tsne: Plot the results of t-SNE dimensionality reduction.
    - plot_umap: Plot the results of UMAP dimensionality reduction.
    - plot_clusters: Plot data points color-coded by cluster labels.
    - plot_interactive_histogram: Create an interactive histogram using Plotly.
    - plot_interactive_correlation: Create an interactive correlation heatmap using Plotly.
    - plot_interactive_scatter: Create an interactive scatter plot using Plotly.
    - plot_feature_importance: Plot feature importance from a machine learning model.
    - plot_barplot: Create a barplot for aggregated numerical values across categories.
    - plot_boxplot_categorical: Create a boxplot for numerical distribution across categories.
    - plot_categorical_distribution: Plot the distribution of a categorical feature.
    - plot_categorical_heatmap: Create a heatmap for co-occurrences between two categorical features.
    - plot_target_distribution: Plot the distribution of a target variable.
    """

    def __init__(self) -> None:
        """
        Initializes the DataVisualizer class.
        """
        pass

    # 1. General Data Exploration
    def plot_distribution(self, df: pd.DataFrame, columns: List[str], kind: str = 'histogram') -> None:
        """
        Plot the distribution of specified columns in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str]): List of column names to plot.
            kind (str): Type of plot ('histogram', 'kde', or 'box'). Default is 'histogram'.

        Raises:
            ValueError: If an unsupported plot kind is provided.
        """
        for col in columns:
            plt.figure(figsize=(8, 4))
            if kind == 'histogram':
                sns.histplot(df[col].dropna(), kde=True)
            elif kind == 'kde':
                sns.kdeplot(df[col].dropna(), shade=True)
            elif kind == 'box':
                sns.boxplot(x=df[col].dropna())
            else:
                raise ValueError(f"Unsupported kind: {kind}. Use 'histogram', 'kde', or 'box'.")
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

    def plot_missing_data(self, df: pd.DataFrame) -> None:
        """
        Visualize missing data in the dataframe using a heatmap.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Data Heatmap")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame, method: str = 'pearson') -> None:
        """
        Plot a heatmap of correlations between numerical features in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            method (str): Correlation method ('pearson', 'spearman', 'kendall'). Default is 'pearson'.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr(method=method)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title(f"Correlation Heatmap ({method.capitalize()})")
        plt.show()

    def plot_swarmplot(self, df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> None:
        """
        Create a swarmplot to visualize the distribution of data points across different categories.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (str): The categorical feature to plot on the x-axis.
            y (str): The numerical feature to plot on the y-axis.
            hue (str, optional): Column name for adding a hue to the plot.

        Raises:
            ValueError: If x or y is not in the dataframe columns.
        """
        if x not in df.columns:
            raise ValueError(f"Column '{x}' not found in dataframe.")
        if y not in df.columns:
            raise ValueError(f"Column '{y}' not found in dataframe.")
        plt.figure(figsize=(10, 6))
        sns.swarmplot(x=x, y=y, hue=hue, data=df)
        plt.title(f'Swarmplot of {y} by {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_3d_scatter(self, df: pd.DataFrame, x: str, y: str, z: str, color: Optional[str] = None) -> None:
        """
        Create a 3D scatter plot for visualizing relationships between three numerical features.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (str): X-axis column.
            y (str): Y-axis column.
            z (str): Z-axis column.
            color (str, optional): Column for coloring the points.

        Raises:
            ValueError: If x, y, or z is not in the dataframe columns.
        """
        for col in [x, y, z]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=f"3D Scatter Plot of {x}, {y}, {z}")
        fig.show()

    # 2. Feature Interactions
    def plot_pairwise_relationships(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Plot pairwise relationships between features.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str]): List of column names to plot pairwise relationships.

        Raises:
            ValueError: If any column in columns is not in the dataframe.
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in dataframe.")
        sns.pairplot(df[columns].dropna(), diag_kind="kde")
        plt.show()

    def plot_scatter_with_outliers(self, df: pd.DataFrame, x: str, y: str, outliers: pd.Series) -> None:
        """
        Plot scatter plot with outliers highlighted.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (str): Name of the x-axis feature.
            y (str): Name of the y-axis feature.
            outliers (pd.Series): Boolean series indicating outliers.

        Raises:
            ValueError: If x, y, or outliers are not valid.
        """
        if x not in df.columns:
            raise ValueError(f"Column '{x}' not found in dataframe.")
        if y not in df.columns:
            raise ValueError(f"Column '{y}' not found in dataframe.")
        if not isinstance(outliers, pd.Series):
            raise ValueError("outliers must be a pandas Series.")
        if len(df) != len(outliers):
            raise ValueError("Length of outliers Series must match length of dataframe.")
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x], df[y], c=outliers.map({True: 'red', False: 'blue'}), edgecolor='k', alpha=0.7)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Scatter plot of {x} vs {y} with Outliers Highlighted')
        plt.show()

    # 3. Outlier Detection Visualization
    def plot_boxplot_with_outliers(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Plot boxplots for columns to visualize potential outliers.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str]): List of column names to plot.

        Raises:
            ValueError: If any column in columns is not in the dataframe.
        """
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in dataframe.")
        plt.figure(figsize=(12, 6))
        df[columns].boxplot()
        plt.title("Box Plot for Outlier Detection")
        plt.ylabel("Value")
        plt.show()

    def plot_isolation_forest_outliers(self, df: pd.DataFrame, outliers: pd.Series) -> None:
        """
        Highlight outliers detected by Isolation Forest in a scatter plot.

        Args:
            df (pd.DataFrame): Input dataframe (should have at least two columns).
            outliers (pd.Series): Boolean series indicating outliers.

        Raises:
            ValueError: If df does not have at least two numeric columns or outliers is not valid.
        """
        if df.select_dtypes(include=[np.number]).shape[1] < 2:
            raise ValueError("Dataframe must have at least two numeric columns.")
        if not isinstance(outliers, pd.Series):
            raise ValueError("outliers must be a pandas Series.")
        if len(df) != len(outliers):
            raise ValueError("Length of outliers Series must match length of dataframe.")
        numeric_df = df.select_dtypes(include=[np.number])
        x_col, y_col = numeric_df.columns[:2]
        fig = px.scatter(df, x=x_col, y=y_col, color=outliers.map({True: 'Outlier', False: 'Inlier'}))
        fig.update_layout(title='Isolation Forest Outliers', xaxis_title=x_col, yaxis_title=y_col)
        fig.show()

    # 4. Temporal Data Visualization
    def plot_time_series(self, df: pd.DataFrame, date_col: str, value_col: str, rolling_window: Optional[int] = None) -> None:
        """
        Plot time series data with an optional rolling window.

        Args:
            df (pd.DataFrame): Input dataframe.
            date_col (str): Name of the datetime column.
            value_col (str): Name of the value column to plot.
            rolling_window (int, optional): Optional rolling window size.

        Raises:
            ValueError: If date_col or value_col is not in the dataframe columns.
        """
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in dataframe.")
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in dataframe.")
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[value_col], label='Original Data')
        if rolling_window:
            plt.plot(df[date_col], df[value_col].rolling(window=rolling_window).mean(), label=f'Rolling Mean ({rolling_window})')
        plt.xlabel('Date')
        plt.ylabel(value_col)
        plt.title(f'Time Series of {value_col}')
        plt.legend()
        plt.show()

    # 5. Dimensionality Reduction Visualization
    def plot_pca(self, df: pd.DataFrame, n_components: int = 2, color: Optional[str] = None) -> None:
        """
        Plot the results of Principal Component Analysis (PCA).

        Args:
            df (pd.DataFrame): Input dataframe.
            n_components (int): Number of components to reduce to. Default is 2.
            color (str, optional): Column name to use for coloring the points.

        Raises:
            ValueError: If n_components is less than 1 or greater than the number of numeric features.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if n_components < 1 or n_components > numeric_df.shape[1]:
            raise ValueError(f"n_components must be between 1 and {numeric_df.shape[1]}")
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_df)
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        if color and color in df.columns:
            pca_df[color] = df[color].values
        plt.figure(figsize=(8, 6))
        if n_components == 2:
            sns.scatterplot(x='PC1', y='PC2', hue=color, data=pca_df)
            plt.title('PCA Result')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend()
            plt.show()
        elif n_components == 3:
            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color=color, title='PCA Result')
            fig.show()
        else:
            raise ValueError("n_components must be 2 or 3 for plotting.")

    def plot_tsne(self, df: pd.DataFrame, n_components: int = 2, perplexity: int = 30, color: Optional[str] = None) -> None:
        """
        Plot the results of t-SNE dimensionality reduction.

        Args:
            df (pd.DataFrame): Input dataframe.
            n_components (int): Number of components to reduce to. Default is 2.
            perplexity (int): Perplexity parameter for t-SNE. Default is 30.
            color (str, optional): Column name to use for coloring the points.

        Raises:
            ValueError: If n_components is not 2.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if n_components != 2:
            raise ValueError("n_components must be 2 for t-SNE plotting.")
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        tsne_result = tsne.fit_transform(numeric_df)
        tsne_df = pd.DataFrame(tsne_result, columns=['Component 1', 'Component 2'])
        if color and color in df.columns:
            tsne_df[color] = df[color].values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Component 1', y='Component 2', hue=color, data=tsne_df)
        plt.title('t-SNE Result')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()

    def plot_umap(self, df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, color: Optional[str] = None) -> None:
        """
        Plot the results of UMAP dimensionality reduction.

        Args:
            df (pd.DataFrame): Input dataframe.
            n_components (int): Number of components to reduce to. Default is 2.
            n_neighbors (int): The size of the local neighborhood.
            min_dist (float): Minimum distance between points in the low-dimensional space.
            color (str, optional): Column name to use for coloring the points.

        Raises:
            ValueError: If n_components is not 2.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if n_components != 2:
            raise ValueError("n_components must be 2 for UMAP plotting.")
        # UMAP model
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        umap_result = umap_model.fit_transform(numeric_df)
        umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
        if color and color in df.columns:
            umap_df[color] = df[color].values
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue=color, data=umap_df)
        plt.title('UMAP Result')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        plt.show()

    def plot_clusters(self, df: pd.DataFrame, cluster_labels: pd.Series, method: str = 'pca', n_components: int = 2) -> None:
        """
        Plot data points color-coded by cluster labels using dimensionality reduction.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            cluster_labels (pd.Series): The cluster labels for each data point.
            method (str): The dimensionality reduction method ('pca', 'umap', 'tsne', or 'identity'). Default is 'pca'.
            n_components (int): Number of dimensions to reduce to. Default is 2.

        Raises:
            ValueError: If method is unsupported.
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components)
        elif method == 'identity':
            reducer = None
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        numeric_df = df.select_dtypes(include=[np.number])
        if reducer:
            reduced_data = reducer.fit_transform(numeric_df)
        else:
            reduced_data = numeric_df.values[:, :n_components]

        plot_df = pd.DataFrame(reduced_data, columns=[f'Dim{i+1}' for i in range(n_components)])
        plot_df['Cluster'] = cluster_labels.values

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', palette='tab20', data=plot_df, s=50, edgecolor='k')
        plt.title(f'Clusters Visualized using {method.upper()}')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    # 6. Interactive Visualizations using Plotly
    def plot_interactive_histogram(self, df: pd.DataFrame, column: str) -> None:
        """
        Create an interactive histogram using Plotly.

        Args:
            df (pd.DataFrame): Input dataframe.
            column (str): Column to visualize.

        Raises:
            ValueError: If column is not in dataframe.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")
        fig = px.histogram(df, x=column, nbins=50, title=f'Interactive Histogram of {column}')
        fig.show()

    def plot_interactive_correlation(self, df: pd.DataFrame) -> None:
        """
        Create an interactive correlation heatmap using Plotly.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            zmin=-1,
            zmax=1))
        fig.update_layout(title="Interactive Correlation Heatmap", xaxis_nticks=36)
        fig.show()

    # 7. Interactive Scatter Plots
    def plot_interactive_scatter(self, df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None) -> None:
        """
        Create an interactive scatter plot using Plotly.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (str): X-axis column.
            y (str): Y-axis column.
            color (str, optional): Column for color encoding.
            size (str, optional): Column for size encoding.

        Raises:
            ValueError: If x or y is not in dataframe.
        """
        for col in [x, y]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
        fig = px.scatter(df, x=x, y=y, color=color, size=size, title=f'Interactive Scatter Plot of {x} vs {y}')
        fig.show()

    # 8. Feature Importance Visualization
    def plot_feature_importance(self, feature_importances: np.ndarray, feature_names: List[str]) -> None:
        """
        Plot feature importance from a machine learning model.

        Args:
            feature_importances (np.ndarray): Array of feature importance values.
            feature_names (List[str]): List of feature names.

        Raises:
            ValueError: If lengths of feature_importances and feature_names do not match.
        """
        if len(feature_importances) != len(feature_names):
            raise ValueError("Length of feature_importances and feature_names must match.")
        indices = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(feature_importances)), feature_importances[indices], align='center')
        plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    # 9. Categorical Data Visualization
    def plot_barplot(self, df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> None:
        """
        Create a barplot for visualizing the aggregated values of a numerical feature across categories.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (str): The categorical feature to plot on the x-axis.
            y (str): The numerical feature to aggregate and plot on the y-axis.
            hue (str, optional): Column name for adding a hue to the plot.

        Raises:
            ValueError: If x or y is not in dataframe.
        """
        for col in [x, y]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x, y=y, hue=hue, data=df, ci=None)
        plt.title(f'Barplot of {y} by {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_boxplot_categorical(self, df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> None:
        """
        Create a boxplot to visualize the distribution of a numerical feature across different categories.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (str): The categorical feature to plot on the x-axis.
            y (str): The numerical feature to plot on the y-axis.
            hue (str, optional): Column name for adding a hue to the plot.

        Raises:
            ValueError: If x or y is not in dataframe.
        """
        for col in [x, y]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x, y=y, hue=hue, data=df)
        plt.title(f'Boxplot of {y} by {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def plot_categorical_distribution(self, df: pd.DataFrame, column: str, hue: Optional[str] = None) -> None:
        """
        Plot the distribution of a categorical feature.

        Args:
            df (pd.DataFrame): Input dataframe.
            column (str): Name of the categorical column.
            hue (str, optional): Column name for adding a hue to the plot.

        Raises:
            ValueError: If column is not in dataframe.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, hue=hue, data=df)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    def plot_categorical_heatmap(self, df: pd.DataFrame, col1: str, col2: str) -> None:
        """
        Create a heatmap to visualize the frequency of co-occurrences between two categorical features.

        Args:
            df (pd.DataFrame): Input dataframe.
            col1 (str): Name of the first categorical column.
            col2 (str): Name of the second categorical column.

        Raises:
            ValueError: If col1 or col2 is not in dataframe.
        """
        for col in [col1, col2]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
        crosstab = pd.crosstab(df[col1], df[col2])
        plt.figure(figsize=(10, 6))
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Heatmap of {col1} vs {col2}')
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.show()

    # 10. Plot Target Distribution
    def plot_target_distribution(self, df: pd.DataFrame, target_column: str) -> None:
        """
        Plot the distribution of a target variable (for classification or regression tasks).

        Args:
            df (pd.DataFrame): Input dataframe.
            target_column (str): Name of the target column.

        Raises:
            ValueError: If target_column is not in dataframe.
        """
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataframe.")
        plt.figure(figsize=(8, 6))
        if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
            sns.countplot(x=target_column, data=df)
            plt.ylabel('Count')
        else:
            sns.histplot(df[target_column], kde=True)
            plt.ylabel('Frequency')
        plt.title(f'Target Distribution: {target_column}')
        plt.xlabel(target_column)
        plt.show()
