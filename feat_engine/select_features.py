import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from typing import Union


class FeatureSelector:
    """
    A utility class for selecting important features from datasets using various statistical tests and model-based methods.
    This class provides several techniques, including chi-squared tests, ANOVA F-tests, mutual information, recursive feature elimination (RFE),
    Lasso (L1) regularization, and correlation-based elimination.
    """

    def __init__(self) -> None:
        """
        Initializes the `FeatureSelector` class. No internal state is set by default.
        """
        pass

    @staticmethod
    def select_kbest_chi2(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """
        Select the top k features using the chi-squared statistical test for feature selection.
        This method is best suited for categorical data and non-negative features.

        Args:
            X (pd.DataFrame): The input feature matrix containing independent variables.
            y (pd.Series): The target variable (dependent variable).
            k (int): The number of top features to select (default: 10).

        Returns:
            pd.DataFrame: A DataFrame containing only the top k selected features.
        """
        selector = SelectKBest(chi2, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features)

    @staticmethod
    def select_kbest_anova(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """
        Select the top k features using ANOVA F-test, which evaluates the variance between groups.
        This method works best for continuous numerical features and when the target variable is categorical.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series): The categorical target variable.
            k (int): The number of top features to select (default: 10).

        Returns:
            pd.DataFrame: A DataFrame containing the selected features.
        """
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features)

    @staticmethod
    def select_kbest_mutual_info(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """
        Select the top k features based on mutual information, which measures the dependency between two variables.
        Mutual information is non-parametric and can capture any kind of relationship between features and the target variable.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series): The target variable (categorical or continuous).
            k (int): The number of top features to select (default: 10).

        Returns:
            pd.DataFrame: A DataFrame containing the top k selected features.
        """
        selector = SelectKBest(mutual_info_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features)

    @staticmethod
    def select_rfe(X: pd.DataFrame, y: pd.Series, model: Union[RandomForestClassifier, LogisticRegression] = RandomForestClassifier(), n_features_to_select: int = 10) -> pd.DataFrame:
        """
        Select features using Recursive Feature Elimination (RFE), a method that recursively removes the least important features
        using a given model until the desired number of features is reached.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series): The target variable.
            model (Union[RandomForestClassifier, LogisticRegression]): The machine learning model to use for RFE (default: RandomForestClassifier).
            n_features_to_select (int): The number of features to select (default: 10).

        Returns:
            pd.DataFrame: A DataFrame containing the selected features.
        """
        rfe = RFE(model, n_features_to_select=n_features_to_select)
        X_new = rfe.fit_transform(X, y)
        selected_features = X.columns[rfe.get_support()]
        return pd.DataFrame(X_new, columns=selected_features)

    @staticmethod
    def select_feature_importance(X: pd.DataFrame, y: pd.Series, model: Union[RandomForestClassifier, LogisticRegression, ExtraTreesClassifier] = RandomForestClassifier(), n_features: int = 10) -> pd.DataFrame:
        """
        Select the top n features based on the feature importance scores generated by a model.
        The importance scores can be obtained from tree-based models (e.g., Random Forest, Extra Trees) or logistic regression.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series): The target variable.
            model (Union[RandomForestClassifier, LogisticRegression, ExtraTreesClassifier]): The model to compute feature importance (default: RandomForestClassifier).
            n_features (int): The number of top features to select (default: 10).

        Returns:
            pd.DataFrame: A DataFrame containing the top n selected features.
        """
        model.fit(X, y.values.ravel())
        feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_[0]
        indices = np.argsort(feature_importances)[::-1][:n_features]
        selected_features = X.columns[indices]
        return X[selected_features]

    @staticmethod
    def select_using_variance_threshold(X: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """
        Removes features that have a variance below a specified threshold. Features with low variance are often
        less informative and can be removed to improve model performance.

        Args:
            X (pd.DataFrame): The input feature matrix.
            threshold (float): The variance threshold for removing features (default: 0.1).

        Returns:
            pd.DataFrame: A DataFrame containing features that meet the variance threshold.
        """
        selector = VarianceThreshold(threshold=threshold)
        X_new = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_features)

    @staticmethod
    def select_lasso(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> pd.DataFrame:
        """
        Select features using Lasso (L1) regularization, which applies a penalty to less important features and drives their coefficients to zero.
        This is useful for feature selection when there are many irrelevant features.

        Args:
            X (pd.DataFrame): The input feature matrix.
            y (pd.Series): The target variable.
            alpha (float): The regularization strength (default: 1.0).

        Returns:
            pd.DataFrame: A DataFrame containing the features selected by Lasso.
        """
        lasso = LassoCV(alphas=[alpha], cv=5)
        lasso.fit(X, y)
        selected_features = X.columns[np.abs(lasso.coef_) > 1e-5]
        return X[selected_features]

    @staticmethod
    def select_correlation(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """
        Removes highly correlated features based on a correlation matrix. This method is useful for reducing multicollinearity
        in the dataset by keeping only one feature from pairs of highly correlated features.

        Args:
            X (pd.DataFrame): The input feature matrix.
            threshold (float): The correlation threshold. Features with a correlation above this value will be considered redundant (default: 0.9).

        Returns:
            pd.DataFrame: A DataFrame with correlated features removed.
        """
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

        # Identify features with correlation greater than the threshold
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        X_new = X.drop(columns=to_drop)
        return X_new
