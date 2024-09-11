import pytest
import pandas as pd
from feat_engine.solve_classification import ClassificationSolver
from sklearn.datasets import make_classification
from typing import Tuple, Any


@pytest.fixture
def dummy_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Creates a dummy dataset for testing purposes.
    Returns a DataFrame of features and a Series of target variables.
    """
    X, y = make_classification(n_samples=200, n_features=20, n_informative=15, n_classes=2, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def classification_solver() -> ClassificationSolver:
    """
    Fixture to initialize the ClassificationSolver for testing.
    """
    return ClassificationSolver()


def test_split_data(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test data splitting into training and testing sets.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = classification_solver.split_data(X, y, test_size=0.2)

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(y_train) == 160
    assert len(y_test) == 40


def test_train_model(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test if a model can be successfully trained.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = classification_solver.split_data(X, y, test_size=0.2)

    model = classification_solver.train_model('Logistic Regression', X_train, y_train)
    assert model is not None


def test_evaluate_model(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test model evaluation metrics.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = classification_solver.split_data(X, y, test_size=0.2)

    model = classification_solver.train_model('Random Forest', X_train, y_train)
    evaluation_results = classification_solver.evaluate_model(model, X_test, y_test)

    assert 'accuracy' in evaluation_results
    assert 'precision' in evaluation_results
    assert 'recall' in evaluation_results
    assert 'f1_score' in evaluation_results
    assert evaluation_results['accuracy'] > 0  # Ensure that the accuracy is positive


def test_cross_validate_model(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test cross-validation for a model.
    """
    X, y = dummy_data
    cv_results = classification_solver.cross_validate_model('K-Nearest Neighbors', X, y, cv=3)

    assert 'mean_accuracy' in cv_results
    assert 'std_accuracy' in cv_results
    assert cv_results['mean_accuracy'] > 0


def test_handle_class_imbalance_oversample(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test oversampling using SMOTE for handling class imbalance.
    """
    X, y = dummy_data
    y.iloc[:150] = 0  # Introduce class imbalance
    X_balanced, y_balanced = classification_solver.handle_class_imbalance(X, y, strategy='oversample')

    assert len(X_balanced) > len(X)  # Oversampling should increase the size
    assert len(y_balanced[y_balanced == 1]) == len(y_balanced[y_balanced == 0])  # Classes should be balanced


def test_plot_confusion_matrix(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test confusion matrix plotting.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = classification_solver.split_data(X, y, test_size=0.2)

    model = classification_solver.train_model('Decision Tree', X_train, y_train)
    evaluation_results = classification_solver.evaluate_model(model, X_test, y_test)

    classification_solver.plot_confusion_matrix(evaluation_results['confusion_matrix'], class_names=['Class 0', 'Class 1'])


def test_auto_select_best_model(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Test automatic model selection based on accuracy.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = classification_solver.split_data(X, y, test_size=0.2)

    best_model = classification_solver.auto_select_best_model(X_train, y_train, X_test, y_test)

    assert best_model in classification_solver.models.keys()  # Ensure that the best model is one of the models


def test_save_load_model(classification_solver: ClassificationSolver, dummy_data: Tuple[pd.DataFrame, pd.Series], tmpdir: Any) -> None:
    """
    Test saving and loading of a trained model.
    """
    X, y = dummy_data
    X_train, X_test, y_train, y_test = classification_solver.split_data(X, y, test_size=0.2)

    model = classification_solver.train_model('Gradient Boosting', X_train, y_train)

    # Test saving the model
    model_path = tmpdir.join('model.pkl')
    classification_solver.save_model(model, str(model_path))

    # Test loading the model
    loaded_model = classification_solver.load_model(str(model_path))
    assert loaded_model is not None
