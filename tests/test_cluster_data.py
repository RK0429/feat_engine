import pytest
import pandas as pd
from sklearn.datasets import make_blobs
from feat_engine.cluster_data import DataClustering


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    """
    Fixture to create a dummy dataset with clear clustering for testing.

    Returns:
        pd.DataFrame: A dataframe with synthetic data.
    """
    X, _ = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)
    return pd.DataFrame(X)


@pytest.fixture
def data_clustering() -> DataClustering:
    """
    Fixture to initialize the DataClustering class.

    Returns:
        DataClustering: An instance of the DataClustering class.
    """
    return DataClustering()


def test_kmeans_clustering(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test KMeans clustering and ensure the silhouette score is returned.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels, silhouette_avg = data_clustering.cluster_kmeans(dummy_data, n_clusters=3)

    assert len(cluster_labels) == len(dummy_data)
    assert silhouette_avg > 0  # Ensure silhouette score is positive
    assert len(cluster_labels.unique()) == 3  # Check if 3 clusters were formed


def test_dbscan_clustering(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test DBSCAN clustering and verify cluster labels are returned.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels = data_clustering.cluster_dbscan(dummy_data, eps=0.5, min_samples=5)

    assert len(cluster_labels) == len(dummy_data)
    assert cluster_labels.nunique() > 0  # DBSCAN should form at least one cluster
    assert -1 in cluster_labels.unique()  # DBSCAN should have noise points


def test_agglomerative_clustering(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test Agglomerative clustering and ensure the number of clusters is correct.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels = data_clustering.cluster_agglomerative(dummy_data, n_clusters=3)

    assert len(cluster_labels) == len(dummy_data)
    assert len(cluster_labels.unique()) == 3  # Should form 3 clusters


def test_gmm_clustering(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test Gaussian Mixture Model (GMM) clustering and ensure the silhouette score is returned.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels, silhouette_avg = data_clustering.cluster_gmm(dummy_data, n_components=3)

    assert len(cluster_labels) == len(dummy_data)
    assert silhouette_avg > 0  # Ensure silhouette score is positive
    assert len(cluster_labels.unique()) == 3  # Check if 3 clusters were formed


def test_meanshift_clustering(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test Mean Shift clustering and verify cluster labels are returned.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels = data_clustering.cluster_meanshift(dummy_data)

    assert len(cluster_labels) == len(dummy_data)
    assert cluster_labels.nunique() > 0  # MeanShift should form at least one cluster


def test_optics_clustering(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test OPTICS clustering and ensure the cluster labels are returned.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels = data_clustering.cluster_optics(dummy_data, min_samples=5)

    assert len(cluster_labels) == len(dummy_data)
    assert cluster_labels.nunique() > 0  # OPTICS should form at least one cluster
    assert -1 in cluster_labels.unique()  # OPTICS should have noise points


def test_evaluate_clustering(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test the clustering evaluation using silhouette score.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels, _ = data_clustering.cluster_kmeans(dummy_data, n_clusters=3)
    silhouette_score = data_clustering.evaluate_clustering(dummy_data, cluster_labels, method='silhouette')

    assert silhouette_score > 0  # Silhouette score should be positive


def test_evaluate_clustering_davies_bouldin(data_clustering: DataClustering, dummy_data: pd.DataFrame) -> None:
    """
    Test the clustering evaluation using Davies-Bouldin score.

    Args:
        data_clustering (DataClustering): The instance of the clustering class.
        dummy_data (pd.DataFrame): The dataset to perform clustering on.
    """
    cluster_labels, _ = data_clustering.cluster_kmeans(dummy_data, n_clusters=3)
    db_score = data_clustering.evaluate_clustering(dummy_data, cluster_labels, method='davies_bouldin')

    assert db_score > 0  # Davies-Bouldin score should be positive
