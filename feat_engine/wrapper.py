import os
import pandas as pd
from datetime import datetime
from .solve_classification import ClassificationSolver
from sklearn.base import BaseEstimator


def learn_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame,
    id_name: str,
    pred_name: str,
    search_type: str = 'bayesian',
    n_iter: int = 500,
    output_dir: str = './prediction'
) -> None:
    """
    Learns from the training data and predicts using the test data.

    This function tunes hyperparameters, selects the best model, performs cross-validation,
    and generates predictions on the test set. It saves the predictions to a CSV file
    and also attempts model stacking.

    Parameters
    ----------
    X : pd.DataFrame
        The training features.

    y : pd.Series
        The training target variable.

    test_df : pd.DataFrame
        The test data to make predictions on.

    id_name : str
        The name of the identifier column in the test data for output.

    pred_name : str
        The name of the prediction column for output.

    search_type : str, default='bayesian'
        The hyperparameter search method. Options include 'grid', 'random', or 'bayesian'.

    n_iter : int, default=500
        The number of iterations for hyperparameter tuning.

    output_dir : str, default='./prediction'
        Directory to save prediction CSV files.

    Returns
    -------
    None
    """
    # Ensure the output directory is created
    _ensure_directory(output_dir)

    cs = ClassificationSolver()
    _tune_and_evaluate(cs, X, y, search_type, n_iter, test_df, id_name, pred_name, output_dir)
    _stack_and_predict(cs, X, y, test_df, id_name, pred_name, output_dir)


def _ensure_directory(path: str) -> None:
    """
    Ensures that the specified directory exists, creating it recursively if needed.

    Parameters
    ----------
    path : str
        The path of the directory to ensure exists.

    Returns
    -------
    None
    """
    os.makedirs(path, exist_ok=True)
    print(f"Directory {path} is ready.")


def _tune_and_evaluate(
    cs: ClassificationSolver,
    X: pd.DataFrame,
    y: pd.Series,
    search_type: str,
    n_iter: int,
    test_df: pd.DataFrame,
    id_name: str,
    pred_name: str,
    output_dir: str
) -> None:
    """
    Tunes hyperparameters and evaluates the best model.

    This function performs hyperparameter tuning, selects the best model,
    performs cross-validation, and saves predictions to a CSV file.

    Parameters
    ----------
    cs : ClassificationSolver
        An instance of the ClassificationSolver class.

    X : pd.DataFrame
        The training features.

    y : pd.Series
        The training target variable.

    search_type : str
        The hyperparameter search method.

    n_iter : int
        The number of iterations for hyperparameter tuning.

    test_df : pd.DataFrame
        The test data to make predictions on.

    id_name : str
        The name of the identifier column in the test data for output.

    pred_name : str
        The name of the prediction column for output.

    output_dir : str
        Directory to save prediction CSV files.

    Returns
    -------
    None
    """
    # Hyperparameter tuning
    cs.hyperparameter_tuning("all", X, y, search_type=search_type, n_iter=n_iter)

    # Select best model
    model_name, mean_accuracy = cs.auto_select_best_model(X, y)
    print(f'Model Name: {model_name}, Mean Accuracy: {mean_accuracy}')

    # Cross-validate and predict with the best model
    model = cs.tuned_models[model_name]
    _cross_validate_and_predict(model, cs, X, y, test_df, id_name, pred_name, model_name, output_dir)


def _cross_validate_and_predict(
    model: BaseEstimator,
    cs: ClassificationSolver,
    X: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame,
    id_name: str,
    pred_name: str,
    model_name: str,
    output_dir: str
) -> None:
    """
    Cross-validates a model and makes predictions on test data.

    Parameters
    ----------
    model : any
        The model to evaluate and predict with.

    cs : ClassificationSolver
        An instance of the ClassificationSolver class.

    X : pd.DataFrame
        The training features.

    y : pd.Series
        The training target variable.

    test_df : pd.DataFrame
        The test data to make predictions on.

    id_name : str
        The name of the identifier column in the test data for output.

    pred_name : str
        The name of the prediction column for output.

    model_name : str
        The name of the model being used for prediction.

    output_dir : str
        Directory to save prediction CSV files.

    Returns
    -------
    None
    """
    # Cross-validation
    scores = cs.cross_validate_model(model, X, y)
    print('Scores: ', scores)

    # Make predictions
    pred = model.predict(test_df)
    _save_predictions(test_df, pred, id_name, pred_name, model_name, output_dir)
    print('Prediction: ', pred)


def _save_predictions(
    test_df: pd.DataFrame,
    predictions: pd.Series,
    id_name: str,
    pred_name: str,
    model_name: str,
    output_dir: str
) -> None:
    """
    Saves predictions to a CSV file.

    Parameters
    ----------
    test_df : pd.DataFrame
        The test data including IDs.

    predictions : pd.Series
        The predicted values.

    id_name : str
        The name of the identifier column.

    pred_name : str
        The name of the prediction column.

    model_name : str
        The name of the model used for prediction.

    output_dir : str
        Directory to save prediction CSV files.

    Returns
    -------
    None
    """
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    output = pd.DataFrame({id_name: test_df.index, pred_name: predictions})
    file_path = os.path.join(output_dir, f'{model_name}-submission-{now}.csv')
    output.to_csv(file_path, index=False)
    print(f'Saved predictions to {file_path}')


def _stack_and_predict(
    cs: ClassificationSolver,
    X: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame,
    id_name: str,
    pred_name: str,
    output_dir: str
) -> None:
    """
    Performs model stacking and makes predictions using the stacked model.

    Parameters
    ----------
    cs : ClassificationSolver
        An instance of the ClassificationSolver class.

    X : pd.DataFrame
        The training features.

    y : pd.Series
        The training target variable.

    test_df : pd.DataFrame
        The test data to make predictions on.

    id_name : str
        The name of the identifier column in the test data for output.

    pred_name : str
        The name of the prediction column for output.

    output_dir : str
        Directory to save prediction CSV files.

    Returns
    -------
    None
    """
    print('Merging models...')

    # Perform model stacking
    model = cs.model_merging(cs.models.keys(), X, y)

    # Cross-validation and prediction with the stacked model
    scores = cs.cross_validate_model(model, X, y)
    print('Scores: ', scores)

    # Make predictions with stacked model
    pred = model.predict(test_df)
    _save_predictions(test_df, pred, id_name, pred_name, 'stacked-model', output_dir)
    print('Prediction: ', pred)
