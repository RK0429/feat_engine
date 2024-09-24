import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union

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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.exceptions import NotFittedError

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformer for selecting important features from datasets using various statistical tests
    and model-based methods.

    This class supports multiple feature selection techniques, including:
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
            estimator = self.model or (RandomForestClassifier() if self.problem_type == 'classification' else RandomForestRegressor())
            self.selector_ = RFE(estimator=estimator, n_features_to_select=self.k)
            self.selector_.fit(X, y)
        elif self.method == 'lasso':
            if self.problem_type == 'classification':
                estimator = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=1.0 / self.alpha,
                    random_state=42
                )
            else:
                estimator = Lasso(alpha=self.alpha, random_state=42)
            self.selector_ = SelectFromModel(estimator=estimator)
            self.selector_.fit(X, y)
        elif self.method == 'feature_importance':
            estimator = self.estimator or (RandomForestClassifier() if self.problem_type == 'classification' else RandomForestRegressor())
            self.selector_ = SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=self.k)
            self.selector_.fit(X, y)
        elif self.method == 'correlation':
            self._fit_correlation(X)
            return self
        else:
            raise ValueError(f"Unknown method: {self.method}")

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
        if self.selected_features_ is None or self.selector_ is None:
            raise NotFittedError("This FeatureSelector instance is not fitted yet.")

        if self.method == 'correlation':
            return X[self.selected_features_]

        transformed_X = self.selector_.transform(X)  # type: ignore
        transformed_df = pd.DataFrame(transformed_X, columns=self.selected_features_, index=X.index)
        return transformed_df

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

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int], Any]:
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
            return np.where(self.support_)[0]
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
        if input_features is None:
            raise ValueError("input_features must be provided.")
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


class FeatureSelectorWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for different feature selection methods to be used within a pipeline.

    This class allows dynamic selection and application of various feature selection methods
    based on the provided 'selector_type'.
    """

    def __init__(
        self,
        selector_type: str = 'selectkbest_f_classif',
        score_func: Optional[Any] = None,
        k: int = 10,
        threshold: float = 0.0,
        estimator: Optional[Any] = None,
        n_features_to_select: int = 10,
        alpha: float = 1.0,  # Added alpha parameter
    ) -> None:
        """
        Initialize the FeatureSelectorWrapper.

        Parameters
        ----------
        selector_type : str, default='selectkbest_f_classif'
            Identifier for the feature selection method. Supported types:
            - 'selectkbest_f_classif'
            - 'selectkbest_chi2'
            - 'selectkbest_mutual_info_classif'
            - 'selectkbest_mutual_info_regression'
            - 'variance_threshold'
            - 'rfe'
            - 'selectfrommodel'
            - 'lasso'

        score_func : callable, optional
            Scoring function for SelectKBest methods.

        k : int, default=10
            Number of top features to select for SelectKBest.

        threshold : float, default=0.0
            Threshold for VarianceThreshold or SelectFromModel.

        estimator : estimator object, optional
            Estimator to use for RFE or SelectFromModel.

        n_features_to_select : int, default=10
            Number of features to select for RFE.

        alpha : float, default=1.0
            Regularization strength for Lasso-based selection.
        """
        self.selector_type = selector_type
        self.score_func = score_func
        self.k = k
        self.threshold = threshold
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.alpha = alpha  # Store alpha

        self.selector_: Optional[BaseEstimator] = None

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
            if 'chi2' in self.selector_type:
                if y is None:
                    raise ValueError("y cannot be None for chi2.")
                self.selector_ = SelectKBest(score_func=chi2, k=self.k)
            elif 'f_classif' in self.selector_type:
                if y is None:
                    raise ValueError("y cannot be None for f_classif.")
                self.selector_ = SelectKBest(score_func=f_classif, k=self.k)
            elif 'f_regression' in self.selector_type:
                self.selector_ = SelectKBest(score_func=f_regression, k=self.k)
            elif 'mutual_info_classif' in self.selector_type:
                if y is None:
                    raise ValueError("y cannot be None for mutual_info_classif.")
                self.selector_ = SelectKBest(score_func=mutual_info_classif, k=self.k)
            elif 'mutual_info_regression' in self.selector_type:
                self.selector_ = SelectKBest(score_func=mutual_info_regression, k=self.k)
            else:
                raise ValueError(f"Unknown SelectKBest variant: {self.selector_type}")
            self.selector_.fit(X, y)
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
            if y is None:
                raise ValueError("y cannot be None for lasso.")
            # Initialize Lasso estimator with the provided alpha
            lasso = Lasso(alpha=self.alpha, random_state=42)
            self.selector_ = SelectFromModel(estimator=lasso)
            self.selector_.fit(X, y)
        else:
            raise ValueError(f"Unknown selector type: {self.selector_type}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray | Any:
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        X_transformed : np.ndarray
            The transformed feature matrix containing only the selected features.
        """
        if self.selector_ is None:
            raise NotFittedError("This FeatureSelectorWrapper instance is not fitted yet.")
        return self.selector_.transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int], Any]:
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
            raise NotFittedError("This FeatureSelectorWrapper instance is not fitted yet.")
        return self.selector_.get_support(indices=indices)

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
        if self.selector_ is None:
            raise NotFittedError("This FeatureSelectorWrapper instance is not fitted yet.")
        if input_features is None:
            raise ValueError("input_features must be provided.")
        support = self.get_support(indices=True)
        return [input_features[i] for i in support]


class AutoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Automatically selects the best feature selection method and optimizes its parameters
    using Grid Search, Random Search, or Bayesian Optimization.

    This class integrates multiple feature selection techniques and utilizes hyperparameter
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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoFeatureSelector':
        """
        Fit the AutoFeatureSelector to the data, selecting the best feature selection method and parameters.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        y : pd.Series, optional
            The target variable. Required for supervised feature selection methods.

        Returns
        -------
        self : AutoFeatureSelector
            Fitted transformer.
        """
        if self.problem_type not in {'classification', 'regression'}:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        # Define default machine learning model if not provided
        if self.model is None:
            self.model = (
                RandomForestClassifier(random_state=self.random_state)
                if self.problem_type == 'classification'
                else RandomForestRegressor(random_state=self.random_state)
            )

        # Define the pipeline with FeatureSelectorWrapper
        pipeline = Pipeline([
            ('selector', FeatureSelectorWrapper()),  # Placeholder for feature selector
            ('model', self.model)
        ])

        # Define parameter search space based on search_type
        if self.search_type in {'grid', 'random'}:
            param_grid = self._get_grid_param_grid(X)
        elif self.search_type == 'bayesian':
            param_grid = self._get_bayes_param_search_space(X)
        else:
            raise ValueError("search_type must be 'grid', 'random', or 'bayesian'.")

        # Choose the search method
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
        else:
            raise ValueError("search_type must be 'grid', 'random', or 'bayesian'.")

        # Fit the search object
        search.fit(X, y)

        # Store the best estimator and its parameters
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

        # Perform transformation
        transformed_X = self.best_estimator_.named_steps['selector'].transform(X)

        # Retrieve the selected feature names
        selected_feature_names = self.get_feature_names_out(input_features=X.columns.tolist())

        # Convert the result back to a DataFrame
        transformed_df = pd.DataFrame(transformed_X, columns=selected_feature_names, index=X.index)

        return transformed_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the AutoFeatureSelector and transform the input data to contain only the selected features.

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

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int], Any]:
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
        return self.best_estimator_.named_steps['selector'].get_support(indices=indices)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str] | Any:
        """
        Get output feature names for transformation based on the best estimator.

        Parameters
        ----------
        input_features : List[str], optional
            Input feature names. If None, feature names are taken from the DataFrame columns.

        Returns
        -------
        selected_features : List[str]
            The list of selected feature names.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutoFeatureSelector instance is not fitted yet.")
        if input_features is None:
            raise ValueError("input_features must be provided.")
        return self.best_estimator_.named_steps['selector'].get_feature_names_out(input_features=input_features)

    def _get_grid_param_grid(self, X: pd.DataFrame) -> List[Dict[str, Any]] | List[object]:
        """
        Generate parameter grid for GridSearchCV and RandomizedSearchCV based on the number of features in X.

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
        step = max(1, num_columns // 5)  # Ensure step is at least 1
        k_min = 5
        k_max = num_columns
        k_values = list(range(k_min, k_max + 1, step))  # Inclusive of k_max

        if self.problem_type == 'classification':
            param_grid = [
                # SelectKBest with ANOVA F-test
                {
                    'selector__selector_type': ['selectkbest_f_classif'],
                    'selector__k': k_values,
                },
                # SelectKBest with Mutual Information
                {
                    'selector__selector_type': ['selectkbest_mutual_info_classif'],
                    'selector__k': k_values,
                },
                # Variance Threshold
                {
                    'selector__selector_type': ['variance_threshold'],
                    'selector__threshold': [0.0, 0.01, 0.1],
                },
                # RFE with Logistic Regression
                {
                    'selector__selector_type': ['rfe'],
                    'selector__n_features_to_select': k_values,
                    'selector__estimator': [LogisticRegression(solver='liblinear', random_state=42)],
                },
                # Lasso-based selection
                {
                    'selector__selector_type': ['lasso'],
                    'selector__alpha': [0.1, 1.0, 10.0],
                },
                # Feature Importance from Model
                {
                    'selector__selector_type': ['feature_importance'],
                    'selector__threshold': ['mean', 'median', 0.0],
                },
                # Correlation-based selection (no parameters)
                {
                    'selector__selector_type': ['correlation'],
                },
            ]
        else:
            # Regression
            param_grid = [
                # SelectKBest with F-regression
                {
                    'selector__selector_type': ['selectkbest_f_regression'],
                    'selector__k': k_values,
                },
                # SelectKBest with Mutual Information
                {
                    'selector__selector_type': ['selectkbest_mutual_info_regression'],
                    'selector__k': k_values,
                },
                # Variance Threshold
                {
                    'selector__selector_type': ['variance_threshold'],
                    'selector__threshold': [0.0, 0.01, 0.1],
                },
                # RFE with Lasso
                {
                    'selector__selector_type': ['rfe'],
                    'selector__n_features_to_select': k_values,
                    'selector__estimator': [Lasso(alpha=1.0, random_state=42)],
                },
                # Lasso-based selection
                {
                    'selector__selector_type': ['lasso'],
                    'selector__alpha': [0.1, 1.0, 10.0],
                },
                # Feature Importance from Model
                {
                    'selector__selector_type': ['feature_importance'],
                    'selector__threshold': ['mean', 'median', 0.0],
                },
                # Correlation-based selection (no parameters)
                {
                    'selector__selector_type': ['correlation'],
                },
            ]

        return param_grid

    def _get_bayes_param_search_space(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate parameter search space for Bayesian optimization based on the number of features in X.

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
        k_min = 5
        k_max = num_columns

        if self.problem_type == 'classification':
            search_space = [
                # SelectKBest with ANOVA F-test
                {
                    'selector__selector_type': Categorical(['selectkbest_f_classif']),
                    'selector__k': Integer(k_min, k_max),
                },
                # SelectKBest with Mutual Information
                {
                    'selector__selector_type': Categorical(['selectkbest_mutual_info_classif']),
                    'selector__k': Integer(k_min, k_max),
                },
                # Variance Threshold
                {
                    'selector__selector_type': Categorical(['variance_threshold']),
                    'selector__threshold': Real(0.0, 0.5, prior='uniform'),
                },
                # RFE with Logistic Regression
                {
                    'selector__selector_type': Categorical(['rfe']),
                    'selector__n_features_to_select': Integer(k_min, k_max),
                    'selector__estimator': Categorical([
                        LogisticRegression(solver='liblinear', random_state=42),
                        RandomForestClassifier(random_state=42)
                    ]),
                },
                # Lasso-based selection
                {
                    'selector__selector_type': Categorical(['lasso']),
                    'selector__alpha': Real(0.1, 10.0, prior='log-uniform'),
                },
                # Feature Importance from Model
                {
                    'selector__selector_type': Categorical(['feature_importance']),
                    'selector__threshold': Categorical(['mean', 'median', 0.0]),
                },
                # Correlation-based selection (no parameters)
                {
                    'selector__selector_type': Categorical(['correlation']),
                },
            ]
        else:
            # Regression
            search_space = [
                # SelectKBest with F-regression
                {
                    'selector__selector_type': Categorical(['selectkbest_f_regression']),
                    'selector__k': Integer(k_min, k_max),
                },
                # SelectKBest with Mutual Information
                {
                    'selector__selector_type': Categorical(['selectkbest_mutual_info_regression']),
                    'selector__k': Integer(k_min, k_max),
                },
                # Variance Threshold
                {
                    'selector__selector_type': Categorical(['variance_threshold']),
                    'selector__threshold': Real(0.0, 0.5, prior='uniform'),
                },
                # RFE with Lasso
                {
                    'selector__selector_type': Categorical(['rfe']),
                    'selector__n_features_to_select': Integer(k_min, k_max),
                    'selector__estimator': Categorical([
                        Lasso(alpha=1.0, random_state=42),
                        RandomForestRegressor(random_state=42)
                    ]),
                },
                # Lasso-based selection
                {
                    'selector__selector_type': Categorical(['lasso']),
                    'selector__alpha': Real(0.1, 10.0, prior='log-uniform'),
                },
                # Feature Importance from Model
                {
                    'selector__selector_type': Categorical(['feature_importance']),
                    'selector__threshold': Categorical(['mean', 'median', 0.0]),
                },
                # Correlation-based selection (no parameters)
                {
                    'selector__selector_type': Categorical(['correlation']),
                },
            ]

        return search_space
