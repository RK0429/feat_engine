import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer for selecting important features from datasets using various statistical tests
    and model-based methods.

    Supports multiple feature selection techniques, including:
    - SelectKBest (ANOVA F-test, Chi-Squared, Mutual Information)
    - Variance Threshold
    - Recursive Feature Elimination (RFE)
    - Lasso-based selection
    - Feature Importance from models
    - Correlation-based feature elimination

    Parameters
    ----------
    method : str, default='kbest_anova'
        Feature selection method to use. Supported options:
        - 'kbest_chi2'
        - 'kbest_anova'
        - 'kbest_mutual_info'
        - 'variance_threshold'
        - 'rfe'
        - 'lasso'
        - 'feature_importance'
        - 'correlation'

    k : int, default=10
        Number of top features to select (applicable for k-best methods).

    threshold : float, default=0.0
        Threshold for Variance Threshold method.

    model : estimator object, default=None
        Model to use for model-based selection (e.g., RFE). If None, defaults to
        RandomForestClassifier for classification or RandomForestRegressor for regression.

    estimator : estimator object, default=None
        Estimator to use for SelectFromModel. If None, defaults to
        RandomForestClassifier for classification or RandomForestRegressor for regression.

    scoring : str, default=None
        Scoring function to use. If None, the estimator's default scoring is used.

    alpha : float, default=1.0
        Regularization strength for Lasso.

    corr_threshold : float, default=0.9
        Correlation threshold for correlation-based feature elimination.

    problem_type : str, default='classification'
        Type of problem: 'classification' or 'regression'.

    **kwargs : Any
        Additional keyword arguments.
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
        self.method = method
        self.k = k
        self.threshold = threshold
        self.model = model
        self.estimator = estimator
        self.scoring = scoring
        self.alpha = alpha
        self.corr_threshold = corr_threshold
        self.problem_type = problem_type.lower()
        self.kwargs = kwargs

        self.selector_: Optional[BaseEstimator] = None
        self.support_: Optional[np.ndarray] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Fit the feature selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series, optional
            The target variable. Required for supervised feature selection methods.

        Returns
        -------
        self : FeatureSelector
            Fitted transformer.
        """
        if self.problem_type not in {'classification', 'regression'}:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        if self.method == 'kbest_chi2':
            if self.problem_type != 'classification':
                raise ValueError("Chi-squared test can only be used for classification problems.")
            self.selector_ = SelectKBest(score_func=chi2, k=self.k)
            self.selector_.fit(X, y)
        elif self.method == 'kbest_anova':
            score_func = f_classif if self.problem_type == 'classification' else f_regression
            self.selector_ = SelectKBest(score_func=score_func, k=self.k)
            self.selector_.fit(X, y)
        elif self.method == 'kbest_mutual_info':
            score_func = mutual_info_classif if self.problem_type == 'classification' else mutual_info_regression
            self.selector_ = SelectKBest(score_func=score_func, k=self.k)
            self.selector_.fit(X, y)
        elif self.method == 'variance_threshold':
            self.selector_ = VarianceThreshold(threshold=self.threshold)
            self.selector_.fit(X)
        elif self.method == 'rfe':
            estimator = self.model or (
                RandomForestClassifier(random_state=42) if self.problem_type == 'classification' else RandomForestRegressor(random_state=42)
            )
            self.selector_ = RFE(estimator=estimator, n_features_to_select=self.k)
            self.selector_.fit(X, y)
        elif self.method == 'lasso':
            if self.problem_type == 'classification':
                estimator = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=1.0 / self.alpha,
                    random_state=42,
                )
            else:
                estimator = Lasso(alpha=self.alpha, random_state=42)
            self.selector_ = SelectFromModel(estimator=estimator)
            self.selector_.fit(X, y)
        elif self.method == 'feature_importance':
            estimator = self.estimator or (
                RandomForestClassifier(random_state=42) if self.problem_type == 'classification' else RandomForestRegressor(random_state=42)
            )
            self.selector_ = SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=self.k)
            self.selector_.fit(X, y)
        elif self.method == 'correlation':
            self._fit_correlation(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.method != 'correlation':
            assert self.selector_ is not None
            self.support_ = self.selector_.get_support()
            self.selected_features_ = X.columns[self.support_].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed feature matrix containing only the selected features.
        """
        if self.selected_features_ is None:
            raise NotFittedError("This FeatureSelector instance is not fitted yet.")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series, optional
            The target variable. Required for supervised feature selection methods.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed feature matrix containing only the selected features.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask or integer index of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of indices of the selected features.
            If False, the return value will be a boolean mask.

        Returns
        -------
        support : Union[np.ndarray, List[int]]
            The mask of selected features, or array of indices.
        """
        if self.support_ is None:
            raise NotFittedError("The model has not been fitted yet!")
        if indices:
            return np.where(self.support_)[0]  # type: ignore
        return self.support_

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : List[str], optional
            Input feature names. If None, feature names are taken from the DataFrame columns.

        Returns
        -------
        selected_features : List[str]
            The list of selected feature names.
        """
        if self.selected_features_ is None:
            raise NotFittedError("The model has not been fitted yet!")
        return self.selected_features_

    def _fit_correlation(self, X: pd.DataFrame) -> None:
        """
        Fit the correlation-based feature selector.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.
        """
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > self.corr_threshold)
        ]
        self.selected_features_ = [col for col in X.columns if col not in to_drop]
        self.support_ = X.columns.isin(self.selected_features_)


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on correlation with other features.

    Parameters
    ----------
    threshold : float, default=0.9
        Features with a correlation higher than this threshold will be removed.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.selected_features_: Optional[List[str]] = None
        self.support_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CorrelationSelector':
        """
        Fit the CorrelationSelector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series, optional
            Ignored, exists for compatibility.

        Returns
        -------
        self : CorrelationSelector
            Fitted transformer.
        """
        # Compute the correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]

        # Select features to keep
        self.selected_features_ = [column for column in X.columns if column not in to_drop]

        # Store feature names
        self.feature_names_in_ = X.columns.tolist()

        # Create support mask
        self.support_ = X.columns.isin(self.selected_features_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by selecting non-correlated features.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed data with non-correlated features.
        """
        if self.selected_features_ is None:
            raise NotFittedError("CorrelationSelector has not been fitted yet.")
        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask or integer index of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of indices of the selected features.
            If False, the return value will be a boolean mask.

        Returns
        -------
        support : Union[np.ndarray, List[int]]
            The mask of selected features, or array of indices.
        """
        if self.support_ is None:
            raise NotFittedError("CorrelationSelector has not been fitted yet.")
        if indices:
            return np.where(self.support_)[0]  # type: ignore
        return self.support_


class FeatureSelectorWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for different feature selection methods to be used within a pipeline.

    Allows dynamic selection and application of various feature selection methods
    based on the provided 'selector_type'.
    """

    def __init__(
        self,
        selector_type: str = 'selectkbest_f_classif',
        k: int = 10,
        threshold: Union[float, str] = 0.0,
        estimator: Optional[Any] = None,
        n_features_to_select: int = 10,
        alpha: float = 1.0,  # For Lasso
    ) -> None:
        """
        Initialize the FeatureSelectorWrapper.

        Parameters
        ----------
        selector_type : str, default='selectkbest_f_classif'
            Identifier for the feature selection method.

        k : int, default=10
            Number of top features to select for SelectKBest.

        threshold : float or str, default=0.0
            Threshold for VarianceThreshold or SelectFromModel.

        estimator : estimator object, optional
            Estimator to use for RFE or SelectFromModel.

        n_features_to_select : int, default=10
            Number of features to select for RFE.

        alpha : float, default=1.0
            Regularization strength for Lasso-based selection.
        """
        self.selector_type = selector_type
        self.k = k
        self.threshold = threshold
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.alpha = alpha

        self.selector_: Optional[BaseEstimator] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelectorWrapper':
        """
        Fit the appropriate feature selector based on 'selector_type'.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series, optional
            The target variable.

        Returns
        -------
        self : FeatureSelectorWrapper
            Fitted transformer.
        """
        if self.selector_type.startswith('selectkbest'):
            self._fit_selectkbest(X, y)
        elif self.selector_type == 'variance_threshold':
            self.selector_ = VarianceThreshold(threshold=self.threshold)
            self.selector_.fit(X)
        elif self.selector_type == 'rfe':
            if self.estimator is None:
                raise ValueError("Estimator must be provided for RFE.")
            self.selector_ = RFE(estimator=self.estimator, n_features_to_select=self.n_features_to_select)
            self.selector_.fit(X, y)
        elif self.selector_type == 'selectfrommodel':
            if self.estimator is None:
                raise ValueError("Estimator must be provided for SelectFromModel.")
            self.selector_ = SelectFromModel(estimator=self.estimator, threshold=self.threshold)
            self.selector_.fit(X, y)
        elif self.selector_type == 'lasso':
            self._fit_lasso(X, y)
        elif self.selector_type == 'feature_importance':
            self._fit_feature_importance(X, y)
        elif self.selector_type == 'correlation':
            corr_threshold = self.threshold if isinstance(self.threshold, float) else 0.9
            self.selector_ = CorrelationSelector(threshold=corr_threshold)
            self.selector_.fit(X)
        else:
            raise ValueError(f"Unknown selector type: {self.selector_type}")

        # Get support mask and selected features
        assert self.selector_ is not None
        support = self.selector_.get_support()
        self.selected_features_ = X.columns[support].tolist()

        # Ensure at least one feature is selected
        if not self.selected_features_:
            self._handle_no_features_selected(X, y)

        return self

    def _fit_selectkbest(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """
        Fit SelectKBest based on the selector_type.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series
            The target variable.
        """
        if y is None:
            raise ValueError("y cannot be None for SelectKBest methods.")
        if 'chi2' in self.selector_type:
            self.selector_ = SelectKBest(score_func=chi2, k=self.k)
        elif 'f_classif' in self.selector_type:
            self.selector_ = SelectKBest(score_func=f_classif, k=self.k)
        elif 'f_regression' in self.selector_type:
            self.selector_ = SelectKBest(score_func=f_regression, k=self.k)
        elif 'mutual_info_classif' in self.selector_type:
            self.selector_ = SelectKBest(score_func=mutual_info_classif, k=self.k)
        elif 'mutual_info_regression' in self.selector_type:
            self.selector_ = SelectKBest(score_func=mutual_info_regression, k=self.k)
        else:
            raise ValueError(f"Unknown SelectKBest variant: {self.selector_type}")
        self.selector_.fit(X, y)

    def _fit_lasso(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """
        Fit Lasso-based feature selection.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series
            The target variable.
        """
        if y is None:
            raise ValueError("y cannot be None for lasso.")
        lasso = Lasso(alpha=self.alpha, random_state=42)
        self.selector_ = SelectFromModel(estimator=lasso)
        self.selector_.fit(X, y)

    def _fit_feature_importance(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """
        Fit feature importance-based selection.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series
            The target variable.
        """
        if self.estimator is None:
            self.estimator = RandomForestClassifier(random_state=42)
        self.selector_ = SelectFromModel(estimator=self.estimator, threshold=self.threshold)
        self.selector_.fit(X, y)

    def _handle_no_features_selected(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """
        Handle cases where no features were selected.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series
            The target variable.
        """
        if self.selector_type == 'lasso':
            self.alpha = max(self.alpha / 10, 0.0001)
            self._fit_lasso(X, y)
        elif self.selector_type == 'correlation':
            new_threshold = max(self.threshold / 2, 0.5) if isinstance(self.threshold, float) else 0.5
            self.selector_ = CorrelationSelector(threshold=new_threshold)
            self.selector_.fit(X)
        else:
            # Fallback to SelectKBest with k=1
            self.selector_ = SelectKBest(score_func=f_classif, k=1)
            self.selector_.fit(X, y)

        assert self.selector_ is not None
        support = self.selector_.get_support()
        self.selected_features_ = X.columns[support].tolist()

        if not self.selected_features_:
            raise ValueError(f"Feature selection method '{self.selector_type}' failed to select any features.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed feature matrix containing only the selected features.
        """
        if self.selected_features_ is None:
            raise NotFittedError("This FeatureSelectorWrapper instance is not fitted yet.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series, optional
            The target variable. Required for supervised feature selection methods.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed feature matrix containing only the selected features.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask or integer index of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of indices of the selected features.
            If False, the return value will be a boolean mask.

        Returns
        -------
        support : Union[np.ndarray, List[int]]
            The mask of selected features, or array of indices.
        """
        if self.selector_ is None:
            raise NotFittedError("The model has not been fitted yet!")
        return self.selector_.get_support(indices=indices)  # type: ignore

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : List[str], optional
            Input feature names.

        Returns
        -------
        selected_features : List[str]
            The list of selected feature names.
        """
        if self.selected_features_ is None:
            raise NotFittedError("The model has not been fitted yet!")
        return self.selected_features_


class AutoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Automatically selects the best feature selection method and optimizes its parameters
    using Grid Search, Random Search, or Bayesian Optimization.

    Integrates multiple feature selection techniques and utilizes hyperparameter
    optimization to identify the most effective method and configuration for the given dataset.

    Parameters
    ----------
    problem_type : str, default='classification'
        Type of problem: 'classification' or 'regression'.

    model : estimator object, default=None
        The machine learning model to use. If None, defaults to RandomForestClassifier
        for classification or RandomForestRegressor for regression.

    cv : int, default=5
        Number of cross-validation folds.

    n_iter : int, default=50
        Number of iterations for RandomizedSearchCV or BayesSearchCV.

    scoring : str, optional
        Scoring metric for optimization. If None, the estimator's default scoring is used.

    random_state : int, default=42
        Random seed for reproducibility.

    search_type : str, default='grid'
        Type of hyperparameter search: 'grid', 'random', or 'bayesian'.
    """

    def __init__(
        self,
        problem_type: str = 'classification',
        model: Optional[Any] = None,
        cv: int = 5,
        n_iter: int = 50,
        scoring: Optional[str] = None,
        random_state: int = 42,
        search_type: str = 'grid',  # 'grid', 'random', or 'bayesian'
    ) -> None:
        self.problem_type = problem_type.lower()
        self.model = model
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.search_type = search_type.lower()

        self.best_estimator_: Optional[Pipeline] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoFeatureSelector':
        """
        Fit the AutoFeatureSelector to the data, selecting the best feature selection method and parameters.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series
            The target variable.

        Returns
        -------
        self : AutoFeatureSelector
            Fitted transformer.
        """
        if self.problem_type not in {'classification', 'regression'}:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        if self.model is None:
            self.model = (
                RandomForestClassifier(random_state=self.random_state)
                if self.problem_type == 'classification'
                else RandomForestRegressor(random_state=self.random_state)
            )

        pipeline = Pipeline([
            ('selector', FeatureSelectorWrapper()),
            ('model', self.model),
        ])

        if self.search_type in {'grid', 'random'}:
            param_grid = self._get_grid_param_grid(X)
        elif self.search_type == 'bayesian':
            param_grid = self._get_bayes_param_search_space(X)
        else:
            raise ValueError("search_type must be 'grid', 'random', or 'bayesian'.")

        if self.search_type == 'grid':
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=1,
            )
        elif self.search_type == 'random':
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )
        elif self.search_type == 'bayesian':
            search = BayesSearchCV(
                estimator=pipeline,
                search_spaces=param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_iter=self.n_iter,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )

        search.fit(X, y)

        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data to contain only the selected features based on the best estimator.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed feature matrix containing only the selected features.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutoFeatureSelector instance is not fitted yet.")

        transformed_X = self.best_estimator_.named_steps['selector'].transform(X)
        selected_feature_names = self.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_X, columns=selected_feature_names, index=X.index)

        return transformed_df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the AutoFeatureSelector and transform the input data to contain only the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series
            The target variable.

        Returns
        -------
        X_transformed : pd.DataFrame
            The transformed feature matrix containing only the selected features.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask or integer index of the features selected by the best estimator.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of indices of the selected features.
            If False, the return value will be a boolean mask.

        Returns
        -------
        support : Union[np.ndarray, List[int]]
            The mask of selected features, or array of indices.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutoFeatureSelector instance is not fitted yet.")
        return self.best_estimator_.named_steps['selector'].get_support(indices=indices)  # type: ignore

    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names for transformation based on the best estimator.

        Returns
        -------
        selected_features : List[str]
            The list of selected feature names.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutoFeatureSelector instance is not fitted yet.")
        return self.best_estimator_.named_steps['selector'].get_feature_names_out()  # type: ignore

    def _get_grid_param_grid(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate parameter grid for GridSearchCV and RandomizedSearchCV.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        param_grid : List[Dict[str, Any]]
            The parameter grid.
        """
        num_columns = X.shape[1]
        step = max(1, num_columns // 5)
        k_values = list(range(1, num_columns + 1, step))

        if self.problem_type == 'classification':
            param_grid = [
                {
                    'selector__selector_type': ['selectkbest_f_classif'],
                    'selector__k': k_values,
                },
                {
                    'selector__selector_type': ['selectkbest_mutual_info_classif'],
                    'selector__k': k_values,
                },
                {
                    'selector__selector_type': ['variance_threshold'],
                    'selector__threshold': [0.0, 0.01, 0.1],
                },
                {
                    'selector__selector_type': ['rfe'],
                    'selector__n_features_to_select': k_values,
                    'selector__estimator': [LogisticRegression(solver='liblinear', random_state=42)],
                },
                {
                    'selector__selector_type': ['lasso'],
                    'selector__alpha': [0.001, 0.01, 0.1, 1.0],
                },
                {
                    'selector__selector_type': ['feature_importance'],
                    'selector__threshold': ['mean', 'median', 0.0],
                    'selector__estimator': [RandomForestClassifier(random_state=42)],
                },
                {
                    'selector__selector_type': ['correlation'],
                },
            ]
        else:
            param_grid = [
                {
                    'selector__selector_type': ['selectkbest_f_regression'],
                    'selector__k': k_values,
                },
                {
                    'selector__selector_type': ['selectkbest_mutual_info_regression'],
                    'selector__k': k_values,
                },
                {
                    'selector__selector_type': ['variance_threshold'],
                    'selector__threshold': [0.0, 0.01, 0.1],
                },
                {
                    'selector__selector_type': ['rfe'],
                    'selector__n_features_to_select': k_values,
                    'selector__estimator': [Lasso(alpha=1.0, random_state=42)],
                },
                {
                    'selector__selector_type': ['lasso'],
                    'selector__alpha': [0.001, 0.01, 0.1, 1.0],
                },
                {
                    'selector__selector_type': ['feature_importance'],
                    'selector__threshold': ['mean', 'median', 0.0],
                    'selector__estimator': [RandomForestRegressor(random_state=42)],
                },
                {
                    'selector__selector_type': ['correlation'],
                },
            ]

        return param_grid  # type: ignore

    def _get_bayes_param_search_space(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate parameter search space for Bayesian optimization.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        search_space : List[Dict[str, Any]]
            The parameter search space.
        """
        num_columns = X.shape[1]
        k_min = 1
        k_max = num_columns

        if self.problem_type == 'classification':
            search_space = [
                {
                    'selector__selector_type': Categorical(['selectkbest_f_classif']),
                    'selector__k': Integer(k_min, k_max),
                },
                {
                    'selector__selector_type': Categorical(['selectkbest_mutual_info_classif']),
                    'selector__k': Integer(k_min, k_max),
                },
                {
                    'selector__selector_type': Categorical(['variance_threshold']),
                    'selector__threshold': Real(0.0, 0.5, prior='uniform'),
                },
                {
                    'selector__selector_type': Categorical(['rfe']),
                    'selector__n_features_to_select': Integer(k_min, k_max),
                    'selector__estimator': Categorical([
                        LogisticRegression(solver='liblinear', random_state=42),
                        RandomForestClassifier(random_state=42),
                    ]),
                },
                {
                    'selector__selector_type': Categorical(['lasso']),
                    'selector__alpha': Real(0.001, 1.0, prior='log-uniform'),
                },
                {
                    'selector__selector_type': Categorical(['feature_importance']),
                    'selector__threshold': Categorical(['mean', 'median', 0.0]),
                    'selector__estimator': Categorical([
                        RandomForestClassifier(random_state=42),
                        LogisticRegression(solver='liblinear', random_state=42),
                    ]),
                },
                {
                    'selector__selector_type': Categorical(['correlation']),
                    'selector__threshold': Real(0.5, 0.9, prior='uniform'),
                },
            ]
        else:
            search_space = [
                {
                    'selector__selector_type': Categorical(['selectkbest_f_regression']),
                    'selector__k': Integer(k_min, k_max),
                },
                {
                    'selector__selector_type': Categorical(['selectkbest_mutual_info_regression']),
                    'selector__k': Integer(k_min, k_max),
                },
                {
                    'selector__selector_type': Categorical(['variance_threshold']),
                    'selector__threshold': Real(0.0, 0.5, prior='uniform'),
                },
                {
                    'selector__selector_type': Categorical(['rfe']),
                    'selector__n_features_to_select': Integer(k_min, k_max),
                    'selector__estimator': Categorical([
                        Lasso(alpha=1.0, random_state=42),
                        RandomForestRegressor(random_state=42),
                    ]),
                },
                {
                    'selector__selector_type': Categorical(['lasso']),
                    'selector__alpha': Real(0.001, 1.0, prior='log-uniform'),
                },
                {
                    'selector__selector_type': Categorical(['feature_importance']),
                    'selector__threshold': Categorical(['mean', 'median', 0.0]),
                    'selector__estimator': Categorical([
                        RandomForestRegressor(random_state=42),
                        Lasso(alpha=1.0, random_state=42),
                    ]),
                },
                {
                    'selector__selector_type': Categorical(['correlation']),
                    'selector__threshold': Real(0.5, 0.9, prior='uniform'),
                },
            ]

        return search_space
