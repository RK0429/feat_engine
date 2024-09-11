import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import List, Tuple, Dict, Any, Union


class RegressionSolver:
    """
    A comprehensive class for solving regression problems using various machine learning models.
    Includes methods for data preprocessing, model training, evaluation, hyperparameter tuning,
    cross-validation, and model persistence.
    """

    def __init__(self, models: Union[Dict[str, Any], None] = None):
        """
        Initializes the RegressionSolver with a dictionary of models to use.

        Args:
            models (Dict[str, Any]): A dictionary mapping model names to model instances.
        """
        if models is None:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Support Vector Regressor': SVR(),
                'XGBoost': XGBRegressor(),
                'ElasticNet': ElasticNet(),
                'CatBoost': CatBoostRegressor(verbose=0),
                'LightGBM': LGBMRegressor()
            }
        else:
            self.models = models

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data into training and testing sets.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing sets for features and target.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a given regression model.

        Args:
            model_name (str): The name of the model to train.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            Any: The trained model.
        """
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model

    def polynomial_regression(self, X_train: pd.DataFrame, y_train: pd.Series, degree: int = 2) -> Any:
        """
        Trains a polynomial regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            degree (int): The degree of the polynomial features.

        Returns:
            Any: The trained polynomial regression model.
        """
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_poly, y_train)
        return model

    def stack_models(self, X_train: pd.DataFrame, y_train: pd.Series, estimators: List[Tuple[str, Any]], final_estimator: Any) -> Any:
        """
        Trains a stacking regressor that combines multiple base models.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            estimators (List[Tuple[str, Any]]): List of (name, estimator) tuples for base models.
            final_estimator (Any): The final estimator (e.g., LinearRegression).

        Returns:
            Any: The trained stacking model.
        """
        stack_reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
        stack_reg.fit(X_train, y_train)
        return stack_reg

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the model on test data.

        Args:
            model (Any): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics (MSE, MAE, R^2, and Explained Variance).
        """
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)

        return {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2,
            'explained_variance_score': evs
        }

    def evaluate_with_custom_metric(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, scoring_function: Any) -> Dict[str, Any]:
        """
        Evaluates the model using a custom scoring function.

        Args:
            model (Any): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
            scoring_function (Any): A custom scoring function (e.g., mean_squared_error).

        Returns:
            Dict[str, Any]: A dictionary containing the custom score and other metrics.
        """
        predictions = model.predict(X_test)
        custom_score = scoring_function(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return {
            'custom_score': custom_score,
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2
        }

    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, List[Any]], cv: int = 5) -> Any:
        """
        Performs hyperparameter tuning using GridSearchCV.

        Args:
            model_name (str): The name of the model to tune.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            param_grid (Dict[str, List[Any]]): Parameter grid for hyperparameter tuning.
            cv (int): Number of cross-validation folds.

        Returns:
            Any: The best estimator found by GridSearchCV.
        """
        model = self.models[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def auto_select_best_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """
        Automatically selects the best regression model based on R^2 score.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.

        Returns:
            str: The name of the best performing model.
        """
        best_score = -np.inf
        best_model_name = ""
        for model_name in self.models:
            model = self.train_model(model_name, X_train, y_train)
            score = self.evaluate_model(model, X_test, y_test)['r2_score']
            if score > best_score:
                best_score = score
                best_model_name = model_name
        return best_model_name

    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Cross-validates the model using the specified number of folds.

        Args:
            model_name (str): The name of the model to cross-validate.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            cv (int): Number of cross-validation folds.

        Returns:
            Dict[str, float]: Cross-validation scores (mean and standard deviation of R^2 score).
        """
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return {
            'mean_r2_score': float(np.mean(scores)),
            'std_r2_score': float(np.std(scores))
        }

    def plot_residuals(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plots residuals of the regression model.

        Args:
            model (Any): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        predictions = model.predict(X_test)
        residuals = y_test - predictions

        plt.figure(figsize=(10, 6))
        sns.residplot(x=predictions, y=residuals, lowess=True, color="g")
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()

    def plot_residual_distribution(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plots the distribution of residuals (prediction errors).

        Args:
            model (Any): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        predictions = model.predict(X_test)
        residuals = y_test - predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color="blue")
        plt.title('Residual Distribution')
        plt.xlabel('Residuals')
        plt.show()

    def plot_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """
        Plots feature importance for tree-based models.

        Args:
            model (Any): The trained model.
            feature_names (List[str]): List of feature names.
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()
        else:
            print("Model does not have feature importances.")

    def save_model(self, model: Any, filename: str) -> None:
        """
        Saves the trained model to disk.

        Args:
            model (Any): The trained model.
            filename (str): The path and filename to save the model.
        """
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str) -> Any:
        """
        Loads a trained model from disk.

        Args:
            filename (str): The path and filename to load the model from.

        Returns:
            Any: The loaded model.
        """
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
