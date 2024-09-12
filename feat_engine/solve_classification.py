import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import List, Tuple, Dict, Any, Union


class ClassificationSolver:
    """
    A comprehensive class for solving classification problems using various machine learning models.
    Includes methods for data preprocessing, model training, evaluation, hyperparameter tuning,
    cross-validation, and model persistence.
    """
    def __init__(self, models: Union[Dict[str, Any], None] = None) -> None:
        """
        Initializes the ClassificationSolver with a dictionary of models to use.

        Args:
            models (Dict[str, Any]): A dictionary mapping model names to model instances.
        """
        self.logger = self.setup_logger()
        self.models = models or self.default_models()

    def default_models(self) -> Dict[str, Any]:
        """
        Provides default models for classification tasks.

        Returns:
            Dict[str, Any]: A dictionary of default models.
        """
        return {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Support Vector Machine': SVC(probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'XGBoost': XGBClassifier(),
            'LightGBM': LGBMClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0),
            'Naive Bayes': GaussianNB(),
            'Voting Classifier': VotingClassifier(estimators=[
                ('lr', LogisticRegression()),
                ('rf', RandomForestClassifier()),
                ('svc', SVC(probability=True))
            ], voting='soft')
        }

    @staticmethod
    def setup_logger() -> logging.Logger:
        """
        Sets up a logger for tracking model training and evaluation.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger('ClassificationSolver')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, strategy: str = 'oversample') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handles class imbalance using either oversampling or undersampling.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            strategy (str): Strategy to handle imbalance ('oversample' or 'undersample').

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Balanced features and target.
        """
        if strategy == 'oversample':
            self.logger.info("Applying SMOTE oversampling...")
            oversample = SMOTE()
            X_balanced, y_balanced = oversample.fit_resample(X, y)
        elif strategy == 'undersample':
            self.logger.info("Applying undersampling...")
            undersample = RandomUnderSampler()
            X_balanced, y_balanced = undersample.fit_resample(X, y)
        else:
            raise ValueError("Strategy must be either 'oversample' or 'undersample'.")

        return X_balanced, y_balanced

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
        self.logger.info("Splitting data into training and testing sets...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a given model.

        Args:
            model_name (str): The name of the model to train.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            Any: The trained model.
        """
        model = self.models[model_name]
        self.logger.info(f"Training model: {model_name}...")
        model.fit(X_train, y_train.values.ravel())
        return model

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the model on test data.

        Args:
            model (Any): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        self.logger.info("Evaluating model performance...")
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        return self.get_evaluation_metrics(y_test, predictions, probabilities)

    def get_evaluation_metrics(self, y_test: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Computes evaluation metrics for the model.

        Args:
            y_test (pd.Series): True labels.
            predictions (np.ndarray): Model predictions.
            probabilities (np.ndarray): Model predicted probabilities.

        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics.
        """
        self.logger.info("Computing evaluation metrics...")
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        roc_auc = roc_auc_score(y_test, probabilities) if probabilities is not None else None
        conf_matrix = confusion_matrix(y_test, predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix
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
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        self.logger.info(f"Best parameters found for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def auto_select_best_model(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> str:
        """
        Automatically selects the best model based on cross-validated accuracy score.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            cv (int): Number of cross-validation folds (default: 5).

        Returns:
            str: The name of the best performing model based on cross-validation.
        """
        self.logger.info("Automatically selecting the best model based on cross-validated accuracy...")
        best_score = 0
        best_model_name = ""

        for model_name in self.models:
            self.logger.info(f"Cross-validating model: {model_name}")
            model = self.models[model_name]

            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            std_score = scores.std()

            self.logger.info(f"{model_name} - Mean Accuracy: {mean_score:.4f}, Std: {std_score:.4f}")

            # Check if this model performs better than the current best
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = model_name

        self.logger.info(f"Best model selected: {best_model_name} with cross-validated accuracy: {best_score:.4f}")
        return best_model_name

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, class_names: List[str]) -> None:
        """
        Plots the confusion matrix.

        Args:
            conf_matrix (np.ndarray): The confusion matrix.
            class_names (List[str]): List of class names.
        """
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Plots the ROC curve for the given model.

        Args:
            model (Any): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)[:, 1]
        else:
            probabilities = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, probabilities):.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Cross-validates the model using the specified number of folds.

        Args:
            model_name (str): The name of the model to cross-validate.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            cv (int): Number of cross-validation folds.

        Returns:
            Dict[str, float]: Cross-validation scores (mean and standard deviation of accuracy).
        """
        model = self.models[model_name]
        self.logger.info(f"Cross-validating model: {model_name}")
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores))
        }

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
            self.logger.warning("Model does not have feature importances.")

    def save_model(self, model: Any, filename: str) -> None:
        """
        Saves the trained model to disk.

        Args:
            model (Any): The trained model.
            filename (str): The path and filename to save the model.
        """
        joblib.dump(model, filename)
        self.logger.info(f"Model saved to {filename}")

    def load_model(self, filename: str) -> Any:
        """
        Loads a trained model from disk.

        Args:
            filename (str): The path and filename to load the model from.

        Returns:
            Any: The loaded model.
        """
        model = joblib.load(filename)
        self.logger.info(f"Model loaded from {filename}")
        return model
