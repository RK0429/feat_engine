import pytest
import pandas as pd
import numpy as np
from feat_engine.reduce_dimension import DimensionReducer


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture to provide a sample dataset for testing."""
    data = {
        'x1': [1, 2, 3, 4],
        'x2': [4, 9, 16, 25],
        'x3': [10, 20, 30, 40]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_labels() -> pd.Series:
    """Fixture to provide sample class labels for LDA."""
    return pd.Series([0, 1, 0, 1])


def test_pca(sample_data: pd.DataFrame) -> None:
    """Test Principal Component Analysis (PCA)."""
    dr = DimensionReducer()
    df_pca = dr.pca(sample_data, n_components=2)

    # Verify the result shape
    assert df_pca.shape == (4, 2)
    # Verify PCA result is not null and has variance
    assert np.var(df_pca.values) > 0


def test_lda(sample_data: pd.DataFrame, sample_labels: pd.Series) -> None:
    """Test Linear Discriminant Analysis (LDA)."""
    dr = DimensionReducer()
    df_lda = dr.lda(sample_data, sample_labels, n_components=1)

    # Verify the result shape
    assert df_lda.shape == (4, 1)
    # Verify LDA result is not null and has variance
    assert np.var(df_lda.values) > 0


def test_svd(sample_data: pd.DataFrame) -> None:
    """Test Singular Value Decomposition (SVD)."""
    dr = DimensionReducer()
    df_svd = dr.svd(sample_data, n_components=2)

    # Verify the result shape
    assert df_svd.shape == (4, 2)
    # Verify SVD result is not null and has variance
    assert np.var(df_svd.values) > 0


def test_tsne(sample_data: pd.DataFrame) -> None:
    """Test t-Distributed Stochastic Neighbor Embedding (t-SNE)."""
    dr = DimensionReducer()
    df_tsne = dr.tsne(sample_data, n_components=2, perplexity=2)  # Set perplexity < n_samples
    assert df_tsne.shape == (4, 2)
    assert not df_tsne.isnull().values.any()


def test_umap(sample_data: pd.DataFrame) -> None:
    """Test Uniform Manifold Approximation and Projection (UMAP)."""
    dr = DimensionReducer()
    df_umap = dr.umap(sample_data, n_components=2)

    # Verify the result shape
    assert df_umap.shape == (4, 2)
    # Verify UMAP result is not null
    assert not np.isnan(df_umap.values).any()


def test_autoencoder(sample_data: pd.DataFrame) -> None:
    """Test Autoencoder dimensionality reduction."""
    dr = DimensionReducer()
    df_autoencoder = dr.autoencoder(sample_data, encoding_dim=2, epochs=50)

    # Verify the result shape
    assert df_autoencoder.shape == (4, 2)
    # Verify Autoencoder result is not null
    assert not np.isnan(df_autoencoder.values).any()


def test_factor_analysis(sample_data: pd.DataFrame) -> None:
    """Test Factor Analysis."""
    dr = DimensionReducer()
    df_fa = dr.factor_analysis(sample_data, n_components=2)

    # Verify the result shape
    assert df_fa.shape == (4, 2)
    # Verify Factor Analysis result is not null
    assert not np.isnan(df_fa.values).any()


def test_isomap(sample_data: pd.DataFrame) -> None:
    """Test Isomap."""
    dr = DimensionReducer()
    df_isomap = dr.isomap(sample_data, n_components=2, n_neighbors=2)  # Set n_neighbors < n_samples
    assert df_isomap.shape == (4, 2)
    assert not df_isomap.isnull().values.any()
