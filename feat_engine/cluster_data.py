from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
from typing import Tuple, Any


class DataClustering:
    """
    A class that provides clustering methods for arbitrary data frames using K-means, DBSCAN,
    Agglomerative Clustering, Gaussian Mixture Models, Mean Shift, and OPTICS.
    """

    def cluster_kmeans(self, df: pd.DataFrame, n_clusters: int) -> Tuple[pd.Series, float]:
        """
        Perform k-means clustering on the input data frame.

        Args:
            df (pd.DataFrame): The input data frame containing the features.
            n_clusters (int): The number of clusters to form.

        Returns:
            Tuple[pd.Series, float]: Cluster labels and silhouette score.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(df)

        # Calculate silhouette score to evaluate clustering quality
        silhouette_avg = silhouette_score(df, cluster_labels)

        return pd.Series(cluster_labels, index=df.index), silhouette_avg

    def cluster_dbscan(self, df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.Series:
        """
        Perform DBSCAN clustering on the input data frame.

        Args:
            df (pd.DataFrame): The input data frame containing the features.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

        Returns:
            pd.Series: Cluster labels.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(df)

        return pd.Series(cluster_labels, index=df.index)

    def cluster_agglomerative(self, df: pd.DataFrame, n_clusters: int) -> pd.Series:
        """
        Perform Agglomerative clustering on the input data frame.

        Args:
            df (pd.DataFrame): The input data frame containing the features.
            n_clusters (int): The number of clusters to form.

        Returns:
            pd.Series: Cluster labels.
        """
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agglomerative.fit_predict(df)

        return pd.Series(cluster_labels, index=df.index)

    def cluster_gmm(self, df: pd.DataFrame, n_components: int) -> Tuple[pd.Series, float]:
        """
        Perform Gaussian Mixture Model clustering on the input data frame.

        Args:
            df (pd.DataFrame): The input data frame containing the features.
            n_components (int): The number of mixture components (clusters).

        Returns:
            Tuple[pd.Series, float]: Cluster labels and silhouette score.
        """
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        cluster_labels = gmm.fit_predict(df)

        silhouette_avg = silhouette_score(df, cluster_labels)

        return pd.Series(cluster_labels, index=df.index), silhouette_avg

    def cluster_meanshift(self, df: pd.DataFrame) -> pd.Series:
        """
        Perform Mean Shift clustering on the input data frame.

        Args:
            df (pd.DataFrame): The input data frame containing the features.

        Returns:
            pd.Series: Cluster labels.
        """
        meanshift = MeanShift()
        cluster_labels = meanshift.fit_predict(df)

        return pd.Series(cluster_labels, index=df.index)

    def cluster_optics(self, df: pd.DataFrame, min_samples: int = 5) -> pd.Series:
        """
        Perform OPTICS clustering on the input data frame.

        Args:
            df (pd.DataFrame): The input data frame containing the features.
            min_samples (int): The number of samples required to form a cluster.

        Returns:
            pd.Series: Cluster labels.
        """
        optics = OPTICS(min_samples=min_samples)
        cluster_labels = optics.fit_predict(df)

        return pd.Series(cluster_labels, index=df.index)

    def evaluate_clustering(self, df: pd.DataFrame, cluster_labels: pd.Series, method: str = 'silhouette') -> Any:
        """
        Evaluate the clustering result using the specified method.

        Args:
            df (pd.DataFrame): The input data frame containing the features.
            cluster_labels (pd.Series): The cluster labels assigned to each data point.
            method (str): The evaluation method to use ('silhouette' or 'davies_bouldin').

        Returns:
            Any: The clustering evaluation score.
        """
        if method == 'silhouette':
            return silhouette_score(df, cluster_labels)
        elif method == 'davies_bouldin':
            return davies_bouldin_score(df, cluster_labels)
        else:
            raise ValueError("Unsupported evaluation method. Use 'silhouette' or 'davies_bouldin'.")
