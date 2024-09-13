import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore, mstats
from typing import Optional, Tuple, Any


class ZScoreOutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detects outliers using the Z-Score method.
    """

    def __init__(self, threshold: float = 3.0):
        """
        Initializes the ZScoreOutlierDetector.

        Args:
            threshold (float): Z-score threshold beyond which values are considered outliers (default: 3.0).
        """
        self.threshold = threshold
        self.z_scores_: Optional[pd.DataFrame] = None
        self.outliers_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ZScoreOutlierDetector':
        """
        Calculates Z-scores for the dataset.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            ZScoreOutlierDetector: Fitted detector.
        """
        self.z_scores_ = X.apply(zscore)
        self.outliers_ = (self.z_scores_.abs() > self.threshold)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers from the dataset.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return X[~self.outliers_.any(axis=1)]

    def get_outliers(self) -> pd.DataFrame:
        """
        Returns a boolean DataFrame indicating outliers.

        Returns:
            pd.DataFrame: Boolean DataFrame indicating True for outliers.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return self.outliers_


class IQRBasedOutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detects outliers using the Interquartile Range (IQR) method.
    """

    def __init__(self, factor: float = 1.5):
        """
        Initializes the IQRBasedOutlierDetector.

        Args:
            factor (float): The factor to multiply the IQR by (default: 1.5).
        """
        self.factor = factor
        self.Q1_: Optional[pd.Series] = None
        self.Q3_: Optional[pd.Series] = None
        self.IQR_: Optional[pd.Series] = None
        self.outliers_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IQRBasedOutlierDetector':
        """
        Calculates IQR for the dataset.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            IQRBasedOutlierDetector: Fitted detector.
        """
        self.Q1_ = X.quantile(0.25)
        self.Q3_ = X.quantile(0.75)
        self.IQR_ = self.Q3_ - self.Q1_
        lower_bound = self.Q1_ - self.factor * self.IQR_
        upper_bound = self.Q3_ + self.factor * self.IQR_
        self.outliers_ = (X < lower_bound) | (X > upper_bound)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers from the dataset.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return X[~self.outliers_.any(axis=1)]

    def get_outliers(self) -> pd.DataFrame:
        """
        Returns a boolean DataFrame indicating outliers.

        Returns:
            pd.DataFrame: Boolean DataFrame indicating True for outliers.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return self.outliers_


class IsolationForestOutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detects outliers using the Isolation Forest method.
    """

    def __init__(self, contamination: float = 0.1, random_state: Optional[int] = None, **kwargs: Any):
        """
        Initializes the IsolationForestOutlierDetector.

        Args:
            contamination (float): The proportion of outliers in the data (default: 0.1).
            random_state (int, optional): Random state for reproducibility.
            **kwargs: Additional keyword arguments for sklearn's IsolationForest.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs
        self.model: Optional[IsolationForest] = None
        self.outliers_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IsolationForestOutlierDetector':
        """
        Fits the Isolation Forest model.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            IsolationForestOutlierDetector: Fitted detector.
        """
        self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state, **self.kwargs)
        self.model.fit(X)
        predictions = self.model.predict(X)
        self.outliers_ = pd.Series(predictions == -1, index=X.index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers from the dataset.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return X[~self.outliers_]

    def get_outliers(self) -> pd.Series:
        """
        Returns a boolean Series indicating outliers.

        Returns:
            pd.Series: Boolean Series indicating True for outliers.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return self.outliers_


class DBSCANOutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detects outliers using the DBSCAN method.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs: Any):
        """
        Initializes the DBSCANOutlierDetector.

        Args:
            eps (float): The maximum distance between two samples to be considered as neighbors.
            min_samples (int): The number of samples required to form a dense region.
            **kwargs: Additional keyword arguments for sklearn's DBSCAN.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.kwargs = kwargs
        self.model: Optional[DBSCAN] = None
        self.outliers_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DBSCANOutlierDetector':
        """
        Fits the DBSCAN model.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            DBSCANOutlierDetector: Fitted detector.
        """
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, **self.kwargs)
        self.model.fit(X)
        labels = self.model.labels_
        self.outliers_ = pd.Series(labels == -1, index=X.index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes outliers from the dataset.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return X[~self.outliers_]

    def get_outliers(self) -> pd.Series:
        """
        Returns a boolean Series indicating outliers.

        Returns:
            pd.Series: Boolean Series indicating True for outliers.
        """
        if self.outliers_ is None:
            raise ValueError("The detector has not been fitted yet!")
        return self.outliers_


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Applies Winsorization to limit extreme values in the data.
    """

    def __init__(self, limits: Tuple[float, float] = (0.05, 0.05)):
        """
        Initializes the Winsorizer.

        Args:
            limits (Tuple[float, float]): The fraction of data to Winsorize from the bottom and top (default: (0.05, 0.05)).
        """
        self.limits = limits

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Winsorizer':
        """
        Fits the Winsorizer (no action needed).

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            Winsorizer: Fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Winsorization to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Winsorized DataFrame.
        """
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = mstats.winsorize(X_transformed[col], limits=self.limits)
        return X_transformed


class RobustScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Scales data using the RobustScaler method, which is less sensitive to outliers.
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        copy: bool = True,
        unit_variance: bool = False,
    ):
        """
        Initializes the RobustScalerTransformer.

        Args:
            with_centering (bool): If True, center the data before scaling. Default is True.
            with_scaling (bool): If True, scale the data to interquartile range. Default is True.
            quantile_range (Tuple[float, float]): Quantile range used to calculate scale_. Default is (25.0, 75.0).
            copy (bool): Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array). Default is True.
            unit_variance (bool): If True, scale data so that normally distributed features have a variance of 1. Default is False.
        """
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance
        self.scaler: Optional[RobustScaler] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RobustScalerTransformer':
        """
        Fits the RobustScaler to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            RobustScalerTransformer: Fitted transformer.
        """
        self.scaler = RobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling,
            quantile_range=self.quantile_range,
            copy=self.copy,
            unit_variance=self.unit_variance,
        )
        self.scaler.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the RobustScaler.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        if self.scaler is None:
            raise ValueError("The transformer has not been fitted yet!")
        X_transformed = self.scaler.transform(X)
        return pd.DataFrame(X_transformed, columns=X.columns, index=X.index)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers by setting values beyond a threshold to a maximum or minimum value.
    """

    def __init__(self, method: str = 'iqr', factor: float = 1.5):
        """
        Initializes the OutlierCapper.

        Args:
            method (str): The method to use for capping outliers ('iqr' or 'percentiles'). Default is 'iqr'.
            factor (float): The factor to multiply the IQR by (default: 1.5).
        """
        self.method = method
        self.factor = factor
        self.lower_bounds_: Optional[pd.Series] = None
        self.upper_bounds_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierCapper':
        """
        Calculates the bounds for capping outliers.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            OutlierCapper: Fitted transformer.
        """
        if self.method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_ = Q1 - self.factor * IQR
            self.upper_bounds_ = Q3 + self.factor * IQR
        elif self.method == 'percentiles':
            self.lower_bounds_ = X.quantile(0.01)
            self.upper_bounds_ = X.quantile(0.99)
        else:
            raise ValueError("Unsupported method. Use 'iqr' or 'percentiles'.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Caps outliers in the data.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with outliers capped.
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("The transformer has not been fitted yet!")
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
