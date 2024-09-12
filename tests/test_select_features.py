import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from feat_engine.select_features import FeatureSelector
from typing import Tuple


@pytest.fixture
def dummy_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fixture to create dummy data for testing.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing a DataFrame of features and a Series of target labels.
    """
    X = pd.DataFrame({
        'feature1': np.random.randint(0, 100, 100),
        'feature2': np.random.randint(0, 100, 100),
        'feature3': np.random.randint(0, 100, 100),
        'feature4': np.random.randint(0, 100, 100),
        'feature5': np.random.randint(0, 100, 100)
    })
    y = pd.Series(np.random.randint(0, 2, 100), name="target")
    return X, y


def test_select_kbest_chi2(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection using the chi-squared statistical test.
    """
    X, y = dummy_data
    selected_features = FeatureSelector.select_kbest_chi2(X, y, k=3)

    assert selected_features.shape[1] == 3
    assert isinstance(selected_features, pd.DataFrame)


def test_select_kbest_anova(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection using ANOVA F-test.
    """
    X, y = dummy_data
    selected_features = FeatureSelector.select_kbest_anova(X, y, k=3)

    assert selected_features.shape[1] == 3
    assert isinstance(selected_features, pd.DataFrame)


def test_select_kbest_mutual_info(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection using mutual information.
    """
    X, y = dummy_data
    selected_features = FeatureSelector.select_kbest_mutual_info(X, y, k=3)

    assert selected_features.shape[1] == 3
    assert isinstance(selected_features, pd.DataFrame)


def test_select_rfe_with_random_forest(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection using Recursive Feature Elimination (RFE) with RandomForestClassifier.
    """
    X, y = dummy_data
    selected_features = FeatureSelector.select_rfe(X, y, model=RandomForestClassifier(), n_features_to_select=3)

    assert selected_features.shape[1] == 3
    assert isinstance(selected_features, pd.DataFrame)


def test_select_rfe_with_logistic_regression(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection using Recursive Feature Elimination (RFE) with LogisticRegression.
    """
    X, y = dummy_data
    selected_features = FeatureSelector.select_rfe(X, y, model=LogisticRegression(), n_features_to_select=3)

    assert selected_features.shape[1] == 3
    assert isinstance(selected_features, pd.DataFrame)


def test_select_feature_importance(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection based on model feature importance (RandomForestClassifier).
    """
    X, y = dummy_data
    selected_features = FeatureSelector.select_feature_importance(X, y, model=RandomForestClassifier(), n_features=3)

    assert selected_features.shape[1] == 3
    assert isinstance(selected_features, pd.DataFrame)


def test_select_using_variance_threshold(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection by removing features with low variance.
    """
    X, _ = dummy_data
    X['low_variance'] = 1  # Add a low variance feature
    selected_features = FeatureSelector.select_using_variance_threshold(X, threshold=0.1)

    assert 'low_variance' not in selected_features.columns
    assert isinstance(selected_features, pd.DataFrame)


def test_select_lasso(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection using Lasso (L1) regularization.
    """
    X, y = dummy_data
    selected_features = FeatureSelector.select_lasso(X, y, alpha=1.0)

    assert selected_features.shape[1] > 0  # Ensure some features are selected
    assert isinstance(selected_features, pd.DataFrame)


def test_select_correlation(dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test feature selection by removing highly correlated features.
    """
    X, _ = dummy_data
    X['highly_correlated'] = X['feature1'] * 0.9  # Create a highly correlated feature
    selected_features = FeatureSelector.select_correlation(X, threshold=0.8)

    assert 'highly_correlated' not in selected_features.columns
    assert isinstance(selected_features, pd.DataFrame)
