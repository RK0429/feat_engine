import pytest
import pandas as pd
from feat_engine.solve_regression import RegressionSolver
from sklearn.datasets import make_regression
from typing import Tuple, Any


@pytest.fixture
def dummy_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates a dummy dataset for testing purposes.
    """
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def regression_solver() -> RegressionSolver:
    """
    Fixture to initialize the RegressionSolver for testing.
    """
    return RegressionSolver()


def test_split_data(regression_solver: RegressionSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test data splitting into training and testing sets.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = regression_solver.split_data(X, y, test_size=0.2)

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(y_train) == 160
    assert len(y_test) == 40


def test_train_model(regression_solver: RegressionSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test if a model can be successfully trained.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = regression_solver.split_data(X, y, test_size=0.2)

    model = regression_solver.train_model('Linear Regression', X_train, y_train)
    assert model is not None


def test_evaluate_model(regression_solver: RegressionSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test model evaluation metrics.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = regression_solver.split_data(X, y, test_size=0.2)

    model = regression_solver.train_model('Random Forest', X_train, y_train)
    evaluation_results = regression_solver.evaluate_model(model, X_test, y_test)

    assert 'mean_squared_error' in evaluation_results
    assert 'r2_score' in evaluation_results
    assert evaluation_results['mean_squared_error'] > 0  # Ensure MSE is positive
    assert evaluation_results['r2_score'] <= 1  # R^2 score should be <= 1


def test_cross_validate_model(regression_solver: RegressionSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test cross-validation for a model.
    """
    X, y = dummy_data
    cv_results = regression_solver.cross_validate_model('Ridge', X, y, cv=3)

    assert 'mean_r2_score' in cv_results
    assert 'std_r2_score' in cv_results
    assert cv_results['mean_r2_score'] <= 1  # R^2 score should be <= 1


def test_plot_residuals(regression_solver: RegressionSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test residuals plot.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = regression_solver.split_data(X, y, test_size=0.2)

    model = regression_solver.train_model('Lasso', X_train, y_train)
    regression_solver.plot_residuals(model, X_test, y_test)


def test_auto_select_best_model(regression_solver: RegressionSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test automatic model selection based on R^2 score.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = regression_solver.split_data(X, y, test_size=0.2)

    best_model = regression_solver.auto_select_best_model(X_train, y_train, X_test, y_test)

    assert best_model in regression_solver.models.keys()  # Ensure that the best model is one of the models


def test_save_load_model(regression_solver: RegressionSolver, dummy_data: Tuple[pd.DataFrame, pd.Series], tmpdir: Any) -> None:
    """
    Test saving and loading of a trained model.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = regression_solver.split_data(X, y, test_size=0.2)

    model = regression_solver.train_model('ElasticNet', X_train, y_train)

    # Test saving the model
    model_path = tmpdir.join('model.pkl')
    regression_solver.save_model(model, str(model_path))

    # Test loading the model
    loaded_model = regression_solver.load_model(str(model_path))
    assert loaded_model is not None
