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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Optional, Union, List, Any, Dict
from sklearn.exceptions import NotFittedError


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

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the feature selector and transforms the input data to contain only the selected features.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series, optional): The target variable. Required for supervised feature selection methods.

        Returns:
            pd.DataFrame: The transformed feature matrix containing only the selected features.
        """
        return self.fit(X, y).transform(X)

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


class AutoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that automatically selects the best feature selection method and optimizes its parameters.
    """

    def __init__(
        self,
        problem_type: str = 'classification',
        model: Optional[Any] = None,
        param_distributions: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        n_iter: int = 50,
        scoring: Optional[str] = None,
        random_state: int = 42,
        search_type: str = 'grid',  # 'grid' or 'random'
    ) -> None:
        """
        Initializes the AutomatedFeatureSelector with specified parameters.

        Args:
            problem_type (str): 'classification' or 'regression'. Default is 'classification'.
            model (Any, optional): The machine learning model to use. If None, defaults to RandomForestClassifier or RandomForestRegressor based on problem_type.
            param_distributions (Dict[str, Any], optional): Parameter grid or distributions for hyperparameter optimization.
            cv (int): Number of cross-validation folds. Default is 5.
            n_iter (int): Number of iterations for RandomizedSearchCV. Ignored if search_type is 'grid'.
            scoring (str, optional): Scoring metric for optimization. Default is None.
            random_state (int): Random seed for reproducibility. Default is 42.
            search_type (str): Type of hyperparameter search ('grid' or 'random'). Default is 'grid'.
        """
        self.problem_type = problem_type
        self.model = model
        self.param_distributions = param_distributions
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.search_type = search_type

        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoFeatureSelector':
        """
        Fits the feature selector to the data, automatically selecting the best method and parameters.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series, optional): The target variable. Required for supervised feature selection methods.

        Returns:
            AutomatedFeatureSelector: Returns self.
        """
        if self.problem_type not in ['classification', 'regression']:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        # Define default machine learning model if not provided
        if self.model is None:
            if self.problem_type == 'classification':
                self.model = RandomForestClassifier(random_state=self.random_state)
            else:
                self.model = RandomForestRegressor(random_state=self.random_state)

        # Define the pipeline
        pipe = Pipeline([
            ('selector', 'passthrough'),  # Placeholder for feature selector
            ('model', self.model)
        ])

        # Define parameter grid including different feature selection methods and their parameters
        param_grid = self._get_param_grid(X)

        # Choose the search method
        if self.search_type == 'grid':
            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=1,
            )
        elif self.search_type == 'random':
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'.")

        # Fit the search object
        search.fit(X, y)

        # Store the best estimator and its parameters
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data to contain only the selected features.

        Args:
            X (pd.DataFrame): The input feature matrix.

        Returns:
            pd.DataFrame: The transformed feature matrix containing only the selected features.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutomatedFeatureSelector instance is not fitted yet.")
        return self.best_estimator_.named_steps['selector'].transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the feature selector and transforms the input data to contain only the selected features.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series, optional): The target variable. Required for supervised feature selection methods.

        Returns:
            pd.DataFrame: The transformed feature matrix containing only the selected features.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask, or integer index, of the features selected.

        Args:
            indices (bool): If True, the return value will be an array of indices of the selected features.
                            If False, the return value will be a boolean mask.

        Returns:
            Union[np.ndarray, List[int]]: The mask of selected features, or array of indices.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutomatedFeatureSelector instance is not fitted yet.")
        support = self.best_estimator_.named_steps['selector'].get_support(indices)
        return support

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (List[str], optional): Input feature names. If None, feature names are taken from the DataFrame columns.

        Returns:
            List[str]: The list of selected feature names.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutomatedFeatureSelector instance is not fitted yet.")
        if input_features is None:
            raise ValueError("input_features must be provided.")
        support = self.get_support(indices=True)
        return [input_features[i] for i in support]

    def _get_param_grid(self, X: pd.DataFrame) -> List[Dict[str, Any]] | Dict[str, Any]:
        """
        Generates the parameter grid for hyperparameter optimization based on the number of features in X.

        Args:
            X (pd.DataFrame): The input data to determine the number of features.

        Returns:
            Dict[str, Any]: The parameter grid.
        """
        # Create the dynamic range for 'k' using np.linspace
        num_columns = len(X.columns)
        step = num_columns // 5
        k_values = np.arange(5, num_columns, step, dtype=int)

        # Define default parameter grid for classification and regression problems
        if self.problem_type == 'classification':
            param_grid = [
                # SelectKBest with chi2
                {
                    'selector': [SelectKBest()],
                    'selector__score_func': [chi2],
                    'selector__k': k_values,
                },
                # SelectKBest with f_classif
                {
                    'selector': [SelectKBest()],
                    'selector__score_func': [f_classif],
                    'selector__k': k_values,
                },
                # SelectKBest with mutual_info_classif
                {
                    'selector': [SelectKBest()],
                    'selector__score_func': [mutual_info_classif],
                    'selector__k': k_values,
                },
                # VarianceThreshold
                {
                    'selector': [VarianceThreshold()],
                    'selector__threshold': [0, 0.01, 0.1],
                },
                # Recursive Feature Elimination (RFE) with LogisticRegression
                {
                    'selector': [RFE(estimator=LogisticRegression(solver='liblinear'))],
                    'selector__n_features_to_select': k_values,
                },
                # Recursive Feature Elimination (RFE) with LogisticRegression (L1 normalization)
                {
                    'selector': [RFE(estimator=LogisticRegression(penalty='l1', solver='liblinear'))],
                    'selector__n_features_to_select': k_values,
                },
                # Recursive Feature Elimination (RFE) with RandomForestClassifier
                {
                    'selector': [RFE(estimator=RandomForestClassifier(random_state=self.random_state))],
                    'selector__n_features_to_select': k_values,
                },
                # SelectFromModel with LogisticRegression
                {
                    'selector': [SelectFromModel(LogisticRegression(solver='liblinear'))],
                    'selector__threshold': ['mean', 'median', -np.inf],
                },
                # SelectFromModel with RandomForestClassifier
                {
                    'selector': [SelectFromModel(RandomForestClassifier(random_state=self.random_state))],
                    'selector__threshold': ['mean', 'median', -np.inf],
                },
                # SelectFromModel with GradientBoostingClassifier
                {
                    'selector': [SelectFromModel(estimator=GradientBoostingClassifier(random_state=self.random_state))],
                    'selector__threshold': ['mean', 'median', -np.inf],
                },
            ]
        else:
            # Regression
            param_grid = [
                # SelectKBest with f_regression
                {
                    'selector': [SelectKBest()],
                    'selector__score_func': [f_regression],
                    'selector__k': k_values,
                },
                # SelectKBest with mutual_info_regression
                {
                    'selector': [SelectKBest()],
                    'selector__score_func': [mutual_info_regression],
                    'selector__k': k_values,
                },
                # VarianceThreshold
                {
                    'selector': [VarianceThreshold()],
                    'selector__threshold': [0, 0.01, 0.1],
                },
                # Recursive Feature Elimination (RFE) with Lasso
                {
                    'selector': [RFE(estimator=Lasso(random_state=self.random_state))],
                    'selector__n_features_to_select': k_values,
                },
                # Recursive Feature Elimination (RFE) with RandomForestRegressor
                {
                    'selector': [RFE(estimator=RandomForestRegressor(random_state=self.random_state))],
                    'selector__n_features_to_select': k_values,
                },
                # SelectFromModel with Lasso
                {
                    'selector': [SelectFromModel(Lasso(random_state=self.random_state))],
                    'selector__threshold': ['mean', 'median', -np.inf],
                },
                # SelectFromModel with RandomForestRegressor
                {
                    'selector': [SelectFromModel(RandomForestRegressor(random_state=self.random_state))],
                    'selector__threshold': ['mean', 'median', -np.inf],
                },
                # SelectFromModel with GradientBoostingRegressor
                {
                    'selector': [SelectFromModel(estimator=GradientBoostingRegressor(random_state=self.random_state))],
                    'selector__threshold': ['mean', 'median', -np.inf],
                },
            ]

        return param_grid
