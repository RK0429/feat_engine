import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from typing import Optional, List, Dict, Any


class BaseClustering(BaseEstimator, ClusterMixin):
    """
    Base class for clustering algorithms.
    """

    def __init__(self) -> None:
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseClustering":
        raise NotImplementedError("Subclasses should implement this!")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

    def fit_predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray | None:
        self.fit(X, y)
        return self.labels_

    def evaluate(
        self, X: pd.DataFrame, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the clustering result using the specified metrics.

        Args:
            X (pd.DataFrame): The input data.
            metrics (List[str], optional): The evaluation metrics to use.
                Defaults to ['silhouette', 'davies_bouldin', 'calinski_harabasz'].

        Returns:
            Dict[str, float]: A dictionary of evaluation scores.
        """
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")
        if metrics is None:
            metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
        scores = {}
        for metric in metrics:
            if metric == "silhouette":
                score = silhouette_score(X, self.labels_)
            elif metric == "davies_bouldin":
                score = davies_bouldin_score(X, self.labels_)
            elif metric == "calinski_harabasz":
                score = calinski_harabasz_score(X, self.labels_)
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")
            scores[metric] = score
        return scores


class KMeansClustering(BaseClustering):
    """
    K-Means clustering algorithm.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        algorithm: str = "auto",
    ):
        """
        Initialize KMeans clustering.

        Args:
            n_clusters (int): The number of clusters to form.
            init (str): Method for initialization.
            n_init (int): Number of time the k-means algorithm will be run with different centroid seeds.
            max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
            tol (float): Relative tolerance with regards to inertia to declare convergence.
            random_state (Optional[int]): Determines random number generation for centroid initialization.
            algorithm (str): K-means algorithm to use.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.algorithm = algorithm
        self.model: Optional[KMeans] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "KMeansClustering":
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm=self.algorithm,
        )
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)


class DBSCANClustering(BaseClustering):
    """
    DBSCAN clustering algorithm.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize DBSCAN clustering.

        Args:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            metric (str): The metric to use when calculating distance between instances in a feature array.
            algorithm (str): The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
            leaf_size (int): Leaf size passed to BallTree or cKDTree.
            p (float): The power of the Minkowski metric to be used to calculate distance between points.
            n_jobs (int): The number of parallel jobs to run.
        """
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.model: Optional[DBSCAN] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DBSCANClustering":
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("DBSCAN does not support predict method.")


class AgglomerativeClusteringModel(BaseClustering):
    """
    Agglomerative Clustering algorithm.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        affinity: str = "euclidean",
        linkage: str = "ward",
        distance_threshold: Optional[float] = None,
    ):
        """
        Initialize Agglomerative Clustering.

        Args:
            n_clusters (int): The number of clusters to find.
            affinity (str): Metric used to compute the linkage.
            linkage (str): Which linkage criterion to use.
            distance_threshold (float): The linkage distance threshold above which clusters will not be merged.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.model: Optional[AgglomerativeClustering] = None

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "AgglomerativeClusteringModel":
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
        )
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("AgglomerativeClustering does not support predict method.")


class GaussianMixtureClustering(BaseClustering):
    """
    Gaussian Mixture Model clustering algorithm.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        warm_start: bool = False,
        verbose: int = 0,
    ):
        """
        Initialize Gaussian Mixture Model Clustering.

        Args:
            n_components (int): The number of mixture components.
            covariance_type (str): String describing the type of covariance parameters to use.
            tol (float): Convergence threshold.
            reg_covar (float): Non-negative regularization added to the diagonal of covariance.
            max_iter (int): The number of EM iterations to perform.
            n_init (int): The number of initializations to perform.
            init_params (str): The method used to initialize the weights, the means and the precisions.
            random_state (int): Controls the random seed given to initialization methods.
            warm_start (bool): If 'warm_start' is True, the solution of the last fitting is used as initialization.
            verbose (int): Enable verbose output.
        """
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.model: Optional[GaussianMixture] = None

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "GaussianMixtureClustering":
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
        )
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)


def evaluate_clustering(
    X: pd.DataFrame, labels: np.ndarray, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate clustering performance using specified metrics.

    Args:
        X (pd.DataFrame): The input data.
        labels (np.ndarray): Cluster labels.
        metrics (List[str], optional): List of evaluation metrics.
            Defaults to ['silhouette', 'davies_bouldin', 'calinski_harabasz'].

    Returns:
        Dict[str, float]: Dictionary of evaluation metric scores.
    """
    if metrics is None:
        metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    scores = {}
    for metric in metrics:
        if metric == "silhouette":
            scores["silhouette"] = silhouette_score(X, labels)
        elif metric == "davies_bouldin":
            scores["davies_bouldin"] = davies_bouldin_score(X, labels)
        elif metric == "calinski_harabasz":
            scores["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")
    return scores


def find_optimal_k(
    X: pd.DataFrame, max_k: int = 10, method: str = "silhouette"
) -> Dict[int, float]:
    """
    Find the optimal number of clusters for KMeans clustering.

    Args:
        X (pd.DataFrame): The input data.
        max_k (int): Maximum number of clusters to try.
        method (str): Evaluation metric to use ('silhouette', 'davies_bouldin', 'calinski_harabasz').

    Returns:
        Dict[int, float]: Dictionary mapping number of clusters to evaluation score.
    """
    scores = {}
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        if method == "silhouette":
            score = silhouette_score(X, labels)
        elif method == "davies_bouldin":
            score = davies_bouldin_score(X, labels)
        elif method == "calinski_harabasz":
            score = calinski_harabasz_score(X, labels)
        else:
            raise ValueError(f"Unsupported evaluation metric: {method}")
        scores[k] = score
    return scores
