import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
    VarianceThreshold,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from typing import Optional, Union, List, Any


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A transformer for selecting important features from datasets using various statistical tests and model-based methods.
    This class provides several techniques, including chi-squared tests, ANOVA F-tests, mutual information,
    recursive feature elimination (RFE), Lasso (L1) regularization, and correlation-based elimination.
    """

    def __init__(
        self,
        method: str = 'kbest_anova',
        k: int = 10,
        threshold: float = 0.0,
        model: Optional[Any] = None,
        estimator: Optional[Any] = None,
        scoring: Optional[str] = None,
        alpha: float = 1.0,
        corr_threshold: float = 0.9,
        problem_type: str = 'classification',
        **kwargs: Any,
    ) -> None:
        """
        Initializes the FeatureSelector with the specified method and parameters.

        Args:
            method (str): Feature selection method to use. Options are:
                - 'kbest_chi2'
                - 'kbest_anova'
                - 'kbest_mutual_info'
                - 'variance_threshold'
                - 'rfe'
                - 'lasso'
                - 'feature_importance'
                - 'correlation'
            k (int): Number of top features to select (for k-best methods). Default is 10.
            threshold (float): Threshold for variance threshold method. Default is 0.0.
            model (Any, optional): Model to use for model-based selection (e.g., RFE). If None, defaults to RandomForestClassifier or RandomForestRegressor based on problem_type.
            estimator (Any, optional): Estimator to use for SelectFromModel. If None, defaults to RandomForestClassifier or RandomForestRegressor based on problem_type.
            scoring (str, optional): Scoring function to use. Default is None.
            alpha (float): Regularization strength for Lasso. Default is 1.0.
            corr_threshold (float): Correlation threshold for correlation-based selection. Default is 0.9.
            problem_type (str): 'classification' or 'regression'. Default is 'classification'.
            **kwargs: Additional keyword arguments.
        """
        self.method = method
        self.k = k
        self.threshold = threshold
        self.model = model
        self.estimator = estimator
        self.scoring = scoring
        self.alpha = alpha
        self.corr_threshold = corr_threshold
        self.problem_type = problem_type
        self.kwargs = kwargs

        self.selector: Optional[TransformerMixin] = None
        self.support_: Optional[np.ndarray] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Fits the feature selector to the data.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series, optional): The target variable. Required for supervised feature selection methods.

        Returns:
            FeatureSelector: Returns self.
        """
        if self.problem_type not in ['classification', 'regression']:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        if self.method == 'kbest_chi2':
            if self.problem_type != 'classification':
                raise ValueError("Chi-squared test can only be used for classification problems.")
            self.selector = SelectKBest(score_func=chi2, k=self.k)
            self.selector.fit(X, y)
        elif self.method == 'kbest_anova':
            if self.problem_type == 'classification':
                self.selector = SelectKBest(score_func=f_classif, k=self.k)
            else:
                self.selector = SelectKBest(score_func=f_regression, k=self.k)
            self.selector.fit(X, y)
        elif self.method == 'kbest_mutual_info':
            if self.problem_type == 'classification':
                self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
            else:
                self.selector = SelectKBest(score_func=mutual_info_regression, k=self.k)
            self.selector.fit(X, y)
        elif self.method == 'variance_threshold':
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(X)
        elif self.method == 'rfe':
            if self.model is None:
                self.model = RandomForestClassifier() if self.problem_type == 'classification' else RandomForestRegressor()
            self.selector = RFE(estimator=self.model, n_features_to_select=self.k)
            self.selector.fit(X, y)
        elif self.method == 'lasso':
            if self.problem_type == 'classification':
                estimator = LogisticRegression(penalty='l1', solver='liblinear', C=1.0 / self.alpha)
            else:
                estimator = Lasso(alpha=self.alpha)
            self.selector = SelectFromModel(estimator=estimator)
            self.selector.fit(X, y)
        elif self.method == 'feature_importance':
            if self.estimator is None:
                self.estimator = RandomForestClassifier() if self.problem_type == 'classification' else RandomForestRegressor()
            self.selector = SelectFromModel(estimator=self.estimator, threshold=-np.inf, max_features=self.k)
            self.selector.fit(X, y)
        elif self.method == 'correlation':
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.corr_threshold)]
            self.selected_features_ = [col for col in X.columns if col not in to_drop]
            self.support_ = X.columns.isin(self.selected_features_)
            return self
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.support_ = self.selector.get_support()
        self.selected_features_ = X.columns[self.support_].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data to contain only the selected features.

        Args:
            X (pd.DataFrame): The input feature matrix.

        Returns:
            pd.DataFrame: The transformed feature matrix containing only the selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("The model has not been fitted yet!")
        return X.loc[:, self.selected_features_]

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask, or integer index, of the features selected.

        Args:
            indices (bool): If True, the return value will be an array of indices of the selected features.
                            If False, the return value will be a boolean mask.

        Returns:
            Union[np.ndarray, List[int]]: The mask of selected features, or array of indices.
        """
        if self.support_ is None:
            raise ValueError("The model has not been fitted yet!")
        if indices:
            return np.where(self.support_)[0]  # type: ignore
        else:
            return self.support_

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (List[str], optional): Input feature names. If None, feature names are taken from the DataFrame columns.

        Returns:
            List[str]: The list of selected feature names.
        """
        if self.selected_features_ is None:
            raise ValueError("The model has not been fitted yet!")
        return self.selected_features_
