import pytest
import pandas as pd
import numpy as np
from feat_engine.visualize_data import DataVisualizer
import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide a sample dataframe for testing.
    """
    data = {
        'x1': np.random.normal(size=100),
        'x2': np.random.normal(size=100),
        'y': np.random.randint(0, 2, size=100),
        'category': np.random.choice(['A', 'B', 'C'], size=100),
        'date': pd.date_range(start='2020-01-01', periods=100)
    }
    return pd.DataFrame(data)


def test_plot_distribution(sample_data: pd.DataFrame) -> None:
    """
    Test distribution plotting for various types.
    """
    dv = DataVisualizer()
    try:
        dv.plot_distribution(sample_data, ['x1', 'x2'], kind='histogram')
        dv.plot_distribution(sample_data, ['x1', 'x2'], kind='kde')
        dv.plot_distribution(sample_data, ['x1'], kind='box')
    except Exception as e:
        pytest.fail(f"plot_distribution raised an exception: {e}")


def test_plot_missing_data(sample_data: pd.DataFrame) -> None:
    """
    Test missing data heatmap plotting.
    """
    dv = DataVisualizer()
    sample_data_with_nan = sample_data.copy()
    sample_data_with_nan.iloc[0, 0] = np.nan  # Introduce NaN
    try:
        dv.plot_missing_data(sample_data_with_nan)
    except Exception as e:
        pytest.fail(f"plot_missing_data raised an exception: {e}")


def test_plot_correlation_heatmap(sample_data: pd.DataFrame) -> None:
    """
    Test correlation heatmap plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_correlation_heatmap(sample_data)
    except Exception as e:
        pytest.fail(f"plot_correlation_heatmap raised an exception: {e}")


def test_plot_pairwise_relationships(sample_data: pd.DataFrame) -> None:
    """
    Test pairwise relationships plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_pairwise_relationships(sample_data, ['x1', 'x2'])
    except Exception as e:
        pytest.fail(f"plot_pairwise_relationships raised an exception: {e}")


def test_plot_scatter_with_outliers(sample_data: pd.DataFrame) -> None:
    """
    Test scatter plot with outliers plotting.
    """
    dv = DataVisualizer()
    outliers = sample_data['x1'] > 2  # Mark some data as outliers
    try:
        dv.plot_scatter_with_outliers(sample_data, 'x1', 'x2', outliers)
    except Exception as e:
        pytest.fail(f"plot_scatter_with_outliers raised an exception: {e}")


def test_plot_boxplot_with_outliers(sample_data: pd.DataFrame) -> None:
    """
    Test boxplot with outliers plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_boxplot_with_outliers(sample_data, ['x1', 'x2'])
    except Exception as e:
        pytest.fail(f"plot_boxplot_with_outliers raised an exception: {e}")


def test_plot_isolation_forest_outliers(sample_data: pd.DataFrame) -> None:
    """
    Test isolation forest outliers visualization.
    """
    dv = DataVisualizer()
    outliers = sample_data['x1'] > 2  # Mark some data as outliers
    try:
        dv.plot_isolation_forest_outliers(sample_data, outliers)
    except Exception as e:
        pytest.fail(f"plot_isolation_forest_outliers raised an exception: {e}")


def test_plot_time_series(sample_data: pd.DataFrame) -> None:
    """
    Test time series plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_time_series(sample_data, 'date', 'x1', rolling_window=5)
    except Exception as e:
        pytest.fail(f"plot_time_series raised an exception: {e}")


def test_plot_pca(sample_data: pd.DataFrame) -> None:
    """
    Test PCA result plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_pca(sample_data[['x1', 'x2']], n_components=2)
    except Exception as e:
        pytest.fail(f"plot_pca raised an exception: {e}")


def test_plot_tsne(sample_data: pd.DataFrame) -> None:
    """
    Test t-SNE result plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_tsne(sample_data[['x1', 'x2']], n_components=2, perplexity=30)
    except Exception as e:
        pytest.fail(f"plot_tsne raised an exception: {e}")


def test_plot_interactive_histogram(sample_data: pd.DataFrame) -> None:
    """
    Test interactive histogram plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_interactive_histogram(sample_data, 'x1')
    except Exception as e:
        pytest.fail(f"plot_interactive_histogram raised an exception: {e}")


def test_plot_interactive_correlation(sample_data: pd.DataFrame) -> None:
    """
    Test interactive correlation heatmap plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_interactive_correlation(sample_data)
    except Exception as e:
        pytest.fail(f"plot_interactive_correlation raised an exception: {e}")


def test_plot_interactive_scatter(sample_data: pd.DataFrame) -> None:
    """
    Test interactive scatter plot.
    """
    dv = DataVisualizer()
    try:
        dv.plot_interactive_scatter(sample_data, x='x1', y='x2', color='category')
    except Exception as e:
        pytest.fail(f"plot_interactive_scatter raised an exception: {e}")


def test_plot_feature_importance() -> None:
    """
    Test feature importance plotting.
    """
    dv = DataVisualizer()
    feature_importances = np.random.rand(10)
    feature_names = [f"feature_{i}" for i in range(10)]
    try:
        dv.plot_feature_importance(feature_importances, feature_names)
    except Exception as e:
        pytest.fail(f"plot_feature_importance raised an exception: {e}")


def test_plot_categorical_distribution(sample_data: pd.DataFrame) -> None:
    """
    Test categorical distribution plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_categorical_distribution(sample_data, 'category')
    except Exception as e:
        pytest.fail(f"plot_categorical_distribution raised an exception: {e}")


def test_plot_target_distribution(sample_data: pd.DataFrame) -> None:
    """
    Test target distribution plotting.
    """
    dv = DataVisualizer()
    try:
        dv.plot_target_distribution(sample_data, 'y')
    except Exception as e:
        pytest.fail(f"plot_target_distribution raised an exception: {e}")
