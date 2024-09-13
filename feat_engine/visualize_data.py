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
from itertools import combinations


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
    - display_basic_data: Display basic data such as the number of unique elements in each column and the number of missing values.
    """

    def __init__(self) -> None:
        """
        Initializes the DataVisualizer class.
        """
        pass

    # 1. General Data Exploration
    def plot_distribution(self, df: pd.DataFrame, columns: Optional[List[str]] = None, kind: str = 'histogram') -> None:
        """
        Plot the distribution of specified columns or all possible combinations of columns in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): List of column names to plot. If None, all numeric columns are considered.
            kind (str): Type of plot ('histogram', 'kde', or 'box'). Default is 'histogram'.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]
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

    def plot_swarmplot(self, df: pd.DataFrame, x: Optional[List[str]] = None, y: Optional[List[str]] = None, hue: Optional[str] = None) -> None:
        """
        Create a swarmplot to visualize the distribution of data points across different categories.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (List[str], optional): The categorical feature(s) to plot on the x-axis.
            y (List[str], optional): The numerical feature(s) to plot on the y-axis.
            hue (str, optional): Column name for adding a hue to the plot.
        """
        # Determine categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if x is None:
            x = categorical_cols
        elif isinstance(x, str):
            x = [x]
        if y is None:
            y = numerical_cols
        elif isinstance(y, str):
            y = [y]
        # Generate all possible combinations of x and y
        combinations_xy = [(xi, yi) for xi in x for yi in y]
        for xi, yi in combinations_xy:
            if xi not in df.columns:
                raise ValueError(f"Column '{xi}' not found in dataframe.")
            if yi not in df.columns:
                raise ValueError(f"Column '{yi}' not found in dataframe.")
            plt.figure(figsize=(10, 6))
            sns.swarmplot(x=xi, y=yi, hue=hue, data=df)
            plt.title(f'Swarmplot of {yi} by {xi}')
            plt.xlabel(xi)
            plt.ylabel(yi)
            plt.show()

    def plot_3d_scatter(self, df: pd.DataFrame, x: Optional[List[str]] = None, y: Optional[List[str]] = None, z: Optional[List[str]] = None, color: Optional[str] = None) -> None:
        """
        Create a 3D scatter plot for visualizing relationships between three numerical features.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (List[str], optional): X-axis column(s).
            y (List[str], optional): Y-axis column(s).
            z (List[str], optional): Z-axis column(s).
            color (str, optional): Column for coloring the points.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if x is None:
            x = numeric_cols
        elif isinstance(x, str):
            x = [x]
        if y is None:
            y = numeric_cols
        elif isinstance(y, str):
            y = [y]
        if z is None:
            z = numeric_cols
        elif isinstance(z, str):
            z = [z]
        # Generate all possible combinations of x, y, z
        combinations_xyz = [(xi, yi, zi) for xi in x for yi in y for zi in z if xi != yi and yi != zi and xi != zi]
        for xi, yi, zi in combinations_xyz:
            for col in [xi, yi, zi]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe.")
            fig = px.scatter_3d(df, x=xi, y=yi, z=zi, color=color, title=f"3D Scatter Plot of {xi}, {yi}, {zi}")
            fig.show()

    # 2. Feature Interactions
    def plot_pairwise_relationships(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Plot pairwise relationships between features in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): List of column names to plot pairwise relationships.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        sns.pairplot(df[columns].dropna(), diag_kind="kde")
        plt.show()

    def plot_scatter_with_outliers(self, df: pd.DataFrame, outliers: pd.Series, x: Optional[List[str]] = None, y: Optional[List[str]] = None) -> None:
        """
        Plot scatter plots with outliers highlighted.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (List[str], optional): X-axis column(s).
            y (List[str], optional): Y-axis column(s).
            outliers (pd.Series): Boolean series indicating outliers.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if x is None:
            x = numeric_cols
        elif isinstance(x, str):
            x = [x]
        if y is None:
            y = numeric_cols
        elif isinstance(y, str):
            y = [y]
        combinations_xy = [(xi, yi) for xi in x for yi in y if xi != yi]
        for xi, yi in combinations_xy:
            if xi not in df.columns:
                raise ValueError(f"Column '{xi}' not found in dataframe.")
            if yi not in df.columns:
                raise ValueError(f"Column '{yi}' not found in dataframe.")
            if not isinstance(outliers, pd.Series):
                raise ValueError("outliers must be a pandas Series.")
            if len(df) != len(outliers):
                raise ValueError("Length of outliers Series must match length of dataframe.")
            plt.figure(figsize=(8, 6))
            plt.scatter(df[xi], df[yi], c=outliers.map({True: 'red', False: 'blue'}), edgecolor='k', alpha=0.7)
            plt.xlabel(xi)
            plt.ylabel(yi)
            plt.title(f'Scatter plot of {xi} vs {yi} with Outliers Highlighted')
            plt.show()

    # 3. Outlier Detection Visualization
    def plot_boxplot_with_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Plot boxplots for columns to visualize potential outliers.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): List of column names to plot.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
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
    def plot_time_series(self, df: pd.DataFrame, date_col: Optional[str] = None, value_cols: Optional[List[str]] = None, rolling_window: Optional[int] = None) -> None:
        """
        Plot time series data with an optional rolling window.

        Args:
            df (pd.DataFrame): Input dataframe.
            date_col (str, optional): Name of the datetime column. If None, uses the first datetime column.
            value_cols (List[str], optional): Names of the value columns to plot.
            rolling_window (int, optional): Optional rolling window size.
        """
        if date_col is None:
            date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
            if not date_cols:
                raise ValueError("No datetime column found in dataframe.")
            date_col = date_cols[0]
        else:
            if date_col not in df.columns:
                raise ValueError(f"Column '{date_col}' not found in dataframe.")
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(value_cols, str):
            value_cols = [value_cols]
        for value_col in value_cols:
            if value_col not in df.columns:
                raise ValueError(f"Column '{value_col}' not found in dataframe.")
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
    def plot_pca(self, df: pd.DataFrame, columns: Optional[List[str]] = None, n_components: int = 2, color: Optional[str] = None) -> None:
        """
        Plot the results of Principal Component Analysis (PCA).

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): List of columns to use for PCA. If None, all numeric columns are used.
            n_components (int): Number of components to reduce to. Default is 2.
            color (str, optional): Column name to use for coloring the points.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_df = df[columns].select_dtypes(include=[np.number])
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
    def plot_interactive_histogram(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Create interactive histograms for specified columns.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): List of columns to visualize.
        """
        if columns is None:
            columns = df.columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            fig = px.histogram(df, x=col, nbins=50, title=f'Interactive Histogram of {col}')
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
    def plot_interactive_scatter(self, df: pd.DataFrame, x: Optional[List[str]] = None, y: Optional[List[str]] = None, color: Optional[str] = None, size: Optional[str] = None) -> None:
        """
        Create interactive scatter plots for all possible combinations of x and y columns.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (List[str], optional): X-axis column(s).
            y (List[str], optional): Y-axis column(s).
            color (str, optional): Column for color encoding.
            size (str, optional): Column for size encoding.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if x is None:
            x = numeric_cols
        elif isinstance(x, str):
            x = [x]
        if y is None:
            y = numeric_cols
        elif isinstance(y, str):
            y = [y]
        combinations_xy = [(xi, yi) for xi in x for yi in y if xi != yi]
        for xi, yi in combinations_xy:
            for col in [xi, yi]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe.")
            fig = px.scatter(df, x=xi, y=yi, color=color, size=size, title=f'Interactive Scatter Plot of {xi} vs {yi}')
            fig.show()

    # 8. Feature Importance Visualization
    def plot_feature_importance(self, feature_importances: np.ndarray, feature_names: List[str]) -> None:
        """
        Plot feature importance from a machine learning model.

        Args:
            feature_importances (np.ndarray): Array of feature importance values.
            feature_names (List[str]): List of feature names.
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
    def plot_barplot(self, df: pd.DataFrame, x: Optional[List[str]] = None, y: Optional[List[str]] = None, hue: Optional[str] = None) -> None:
        """
        Create barplots for visualizing the aggregated values of numerical features across categories.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (List[str], optional): The categorical feature(s) to plot on the x-axis.
            y (List[str], optional): The numerical feature(s) to aggregate and plot on the y-axis.
            hue (str, optional): Column name for adding a hue to the plot.
        """
        if x is None:
            x = [col for col in df.columns if df[col].nunique() <= 10]
        elif isinstance(x, str):
            x = [x]
        if y is None:
            y = df.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(y, str):
            y = [y]
        combinations_xy = [(xi, yi) for xi in x for yi in y]
        for xi, yi in combinations_xy:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=xi, y=yi, hue=hue, data=df, ci=None)
            plt.title(f'Barplot of {yi} by {xi}')
            plt.xlabel(xi)
            plt.ylabel(yi)
            plt.show()

    def plot_boxplot_categorical(self, df: pd.DataFrame, x: Optional[List[str]] = None, y: Optional[List[str]] = None, hue: Optional[str] = None, max_unique: int = 10) -> None:
        """
        Create boxplots to visualize the distribution of numerical features across different categories.

        Args:
            df (pd.DataFrame): Input dataframe.
            x (List[str], optional): The categorical feature(s) to plot on the x-axis.
            y (List[str], optional): The numerical feature(s) to plot on the y-axis. If None, only columns with more than 'max_unique' unique elements are considered.
            hue (str, optional): Column name for adding a hue to the plot.
            max_unique (int): Maximum number of unique values to consider a column categorical.
        """
        # Select x columns (categorical)
        if x is None:
            x = [col for col in df.columns if df[col].nunique() <= max_unique]
        elif isinstance(x, str):
            x = [x]

        # Select y columns (numerical with more than max_unique unique values)
        if y is None:
            y = [col for col in df.select_dtypes(include=[np.number]).columns if df[col].nunique() > max_unique]
        elif isinstance(y, str):
            y = [y]

        # Create all combinations of x and y
        combinations_xy = [(xi, yi) for xi in x for yi in y]

        # Plot boxplots
        for xi, yi in combinations_xy:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=xi, y=yi, hue=hue, data=df)
            plt.title(f'Boxplot of {yi} by {xi}')
            plt.xlabel(xi)
            plt.ylabel(yi)
            plt.show()

    def plot_categorical_distribution(self, df: pd.DataFrame, columns: Optional[List[str]] = None, hue: Optional[str] = None, max_unique: int = 10) -> None:
        """
        Plot the distribution of categorical features.

        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): Names of the categorical columns.
            hue (str, optional): Column name for adding a hue to the plot.
            max_unique (int): Maximum number of unique values to consider a column categorical.
        """
        if columns is None:
            columns = [col for col in df.columns if df[col].nunique() <= max_unique]
        elif isinstance(columns, str):
            columns = [columns]
        for col in columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=col, hue=hue, data=df)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()

    def plot_categorical_heatmap(self, df: pd.DataFrame, cols: Optional[List[str]] = None, max_unique: int = 10) -> None:
        """
        Create heatmaps for visualizing the frequency of co-occurrences between categorical features.

        Args:
            df (pd.DataFrame): Input dataframe.
            cols (List[str], optional): List of categorical columns.
            max_unique (int): Maximum number of unique values to consider a column categorical.
        """
        if cols is None:
            cols = [col for col in df.columns if df[col].nunique() <= max_unique]
        elif isinstance(cols, str):
            cols = [cols]
        # Generate all possible combinations of two columns
        combinations_cols = combinations(cols, 2)
        for c1, c2 in combinations_cols:
            crosstab = pd.crosstab(df[c1], df[c2])
            plt.figure(figsize=(10, 6))
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Heatmap of {c1} vs {c2}')
            plt.xlabel(c2)
            plt.ylabel(c1)
            plt.show()

    # 10. Plot Target Distribution
    def plot_target_distribution(self, df: pd.DataFrame, target_columns: Optional[List[str]] = None) -> None:
        """
        Plot the distribution of target variable(s).

        Args:
            df (pd.DataFrame): Input dataframe.
            target_columns (List[str], optional): Names of the target columns.
        """
        if target_columns is None:
            target_columns = [df.columns[-1]]  # Assume the last column is target
        elif isinstance(target_columns, str):
            target_columns = [target_columns]
        for target_column in target_columns:
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

    def display_basic_data(self, df: pd.DataFrame) -> None:
        """
        Display basic data such as the number of unique elements in each column and the number of missing values.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        summary_df = pd.DataFrame({
            'Unique Values': df.nunique(),
            'Missing Values': df.isnull().sum(),
            'Data Type': df.dtypes
        })
        summary_df = summary_df.reset_index().rename(columns={'index': 'Column'})
        print(summary_df)
