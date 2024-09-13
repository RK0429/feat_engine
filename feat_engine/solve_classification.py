import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Any, Dict, List, Optional, Tuple

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    log_loss,
    average_precision_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ClassificationSolver:
    """
    A comprehensive class for solving classification problems using various machine learning models.
    Includes methods for data preprocessing, model training, evaluation, hyperparameter tuning,
    cross-validation, and model persistence.
    """

    def __init__(
        self, models: Optional[Dict[str, BaseEstimator]] = None, random_state: int = 42
    ) -> None:
        """
        Initializes the ClassificationSolver with a dictionary of models to use.

        Args:
            models (Optional[Dict[str, BaseEstimator]]): A dictionary mapping model names to model instances.
            random_state (int): Random seed for reproducibility.
        """
        self.logger = self._setup_logger()
        self.random_state = random_state
        self.models = models or self._default_models()
        self.tuned_models: Dict[str, BaseEstimator] = {}

    def _default_models(self) -> Dict[str, BaseEstimator]:
        """
        Provides default models for classification tasks.

        Returns:
            Dict[str, BaseEstimator]: A dictionary of default models.
        """
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "Support Vector Machine": SVC(
                probability=True, random_state=self.random_state
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(random_state=self.random_state),
            "CatBoost": CatBoostClassifier(
                verbose=0, random_state=self.random_state
            ),
            "Naive Bayes": GaussianNB(),
            "Voting Classifier": VotingClassifier(
                estimators=[
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=1000, random_state=self.random_state
                        ),
                    ),
                    (
                        "rf",
                        RandomForestClassifier(random_state=self.random_state),
                    ),
                    (
                        "svc",
                        SVC(
                            probability=True, random_state=self.random_state
                        ),
                    ),
                ],
                voting="soft",
            ),
        }

    def _default_param_grids(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Provides refined hyperparameter grids for common models to enhance tuning performance.

        Returns:
            Dict[str, Dict[str, List[Any]]]: A dictionary of refined param grids.
        """
        return {
            "Logistic Regression": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs", "saga", "newton-cg"],
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.001, 0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10, 15],
                "subsample": [0.8, 0.9, 1.0],
            },
            "Support Vector Machine": {
                "C": [0.01, 0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "Decision Tree": {
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10, 15],
                "subsample": [0.7, 0.8, 0.9, 1.0],
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                "num_leaves": [31, 50, 100, 150],
                "max_depth": [-1, 10, 20, 30],
            },
            "CatBoost": {
                "iterations": [100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.1, 0.2],
                "depth": [3, 5, 7, 10],
            },
            "Naive Bayes": {
                "var_smoothing": np.logspace(0, -9, num=100),
            },
        }

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """
        Sets up a logger for tracking model training and evaluation.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger("ClassificationSolver")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def handle_class_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = "oversample",
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handles class imbalance using either oversampling or undersampling.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            strategy (str): Strategy to handle imbalance ('oversample', 'undersample', or 'none').

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Balanced features and target.
        """
        random_state = random_state or self.random_state
        if strategy == "oversample":
            self.logger.info("Applying SMOTE oversampling...")
            oversample = SMOTE(random_state=random_state)
            X_balanced, y_balanced = oversample.fit_resample(X, y)
        elif strategy == "undersample":
            self.logger.info("Applying undersampling...")
            undersample = RandomUnderSampler(random_state=random_state)
            X_balanced, y_balanced = undersample.fit_resample(X, y)
        elif strategy == "none":
            self.logger.info("No resampling applied.")
            X_balanced, y_balanced = X.copy(), y.copy()
        else:
            raise ValueError(
                "Strategy must be either 'oversample', 'undersample', or 'none'."
            )

        return X_balanced, y_balanced

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data into training and testing sets.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (Optional[int]): Random seed.
            stratify (Optional[pd.Series]): If not None, data is split in a stratified fashion.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Training and testing sets for features and target.
        """
        random_state = random_state or self.random_state
        self.logger.info("Splitting data into training and testing sets...")
        return train_test_split(  # type: ignore
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_pipeline: bool = False,
    ) -> BaseEstimator:
        """
        Trains a given model.

        Args:
            model_name (str): The name of the model to train.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            use_pipeline (bool): Whether to use a pipeline with scaling.

        Returns:
            BaseEstimator: The trained model.
        """
        if model_name in self.tuned_models:
            model = self.tuned_models[model_name]
            self.logger.info(f"Using tuned model: {model_name}")
        else:
            model = self.models[model_name]
            self.logger.info(f"Training model: {model_name}")

        if use_pipeline:
            self.logger.info("Using pipeline with StandardScaler.")
            pipeline = Pipeline(
                [("scaler", StandardScaler()), ("model", model)]
            )
            pipeline.fit(X_train, y_train)
            return pipeline
        else:
            model.fit(X_train, y_train)
            return model

    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the model on test data.

        Args:
            model (BaseEstimator): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
            class_names (Optional[List[str]]): List of class names.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.
        """
        self.logger.info("Evaluating model performance...")
        predictions = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            probabilities = model.decision_function(X_test)
        else:
            probabilities = None

        metrics = self._get_evaluation_metrics(y_test, predictions, probabilities)
        if class_names:
            self.plot_confusion_matrix(metrics["confusion_matrix"], class_names)
        return metrics

    def _get_evaluation_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes evaluation metrics for the model.

        Args:
            y_true (pd.Series): True labels.
            y_pred (np.ndarray): Model predictions.
            y_proba (Optional[np.ndarray]): Model predicted probabilities.

        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics.
        """
        self.logger.info("Computing evaluation metrics...")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_true, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
        }

        if y_proba is not None:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                # Binary classification
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]
                roc_auc = roc_auc_score(y_true, y_proba)
                average_precision = average_precision_score(y_true, y_proba)
                logloss = log_loss(y_true, y_proba)
                metrics.update(
                    {
                        "roc_auc": roc_auc,
                        "average_precision": average_precision,
                        "log_loss": logloss,
                    }
                )
            else:
                # Multi-class classification
                roc_auc = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
                logloss = log_loss(y_true, y_proba)
                metrics.update({"roc_auc": roc_auc, "log_loss": logloss})

        return metrics

    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
        search_type: str = "grid",
        n_iter: int = 50,
        scoring: str = "accuracy",
    ) -> None:
        """
        Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV for one or all models and stores the best models.

        Args:
            model_name (str): The name of the model to tune. If 'all', tunes all models in self.models.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            param_grid (Optional[Dict[str, List[Any]]]): Parameter grid for hyperparameter tuning. If None, uses default.
            cv (int): Number of cross-validation folds.
            search_type (str): Type of search ('grid' or 'random').
            n_iter (int): Number of iterations for RandomizedSearchCV.
            scoring (str): Scoring metric for evaluation.

        Returns:
            None: The best models are stored in self.tuned_models.
        """
        if model_name == "all":
            self.logger.info("Performing hyperparameter tuning for all models...")
            for name in self.models:
                self._tune_single_model(
                    name, X_train, y_train, param_grid, cv, search_type, n_iter, scoring
                )
        else:
            self._tune_single_model(
                model_name,
                X_train,
                y_train,
                param_grid,
                cv,
                search_type,
                n_iter,
                scoring,
            )

    def _tune_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]],
        cv: int,
        search_type: str,
        n_iter: int,
        scoring: str,
    ) -> None:
        """
        Helper method to perform hyperparameter tuning for a single model.

        Args:
            model_name (str): The name of the model to tune.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            param_grid (Optional[Dict[str, List[Any]]]): Parameter grid for hyperparameter tuning. If None, uses default.
            cv (int): Number of cross-validation folds.
            search_type (str): Type of search ('grid' or 'random').
            n_iter (int): Number of iterations for RandomizedSearchCV.
            scoring (str): Scoring metric for evaluation.

        Returns:
            None: The best model is stored in self.tuned_models.
        """
        model = self.models[model_name]
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")

        if param_grid is None:
            param_grid = self._default_param_grids().get(model_name, {})
            if not param_grid:
                self.logger.warning(
                    f"No parameter grid available for {model_name}. Skipping tuning."
                )
                return

        if search_type == "grid":
            search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state,
            )
        else:
            raise ValueError("search_type must be either 'grid' or 'random'.")

        search.fit(X_train, y_train)
        self.logger.info(
            f"Best parameters found for {model_name}: {search.best_params_}"
        )

        # Store the tuned model for future use
        self.tuned_models[model_name] = search.best_estimator_

    def auto_select_best_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
    ) -> Tuple[str, float]:
        """
        Automatically selects the best model based on cross-validated accuracy score.
        It checks if a hyperparameter-tuned version of the model is available and uses it if present.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            cv (int): Number of cross-validation folds (default: 5).

        Returns:
            Tuple[str, float]: The name of the best performing model and its score based on cross-validation.
        """
        self.logger.info(
            "Automatically selecting the best model based on cross-validated accuracy..."
        )
        best_score = 0.0
        best_model_name = ""

        for model_name in self.models:
            self.logger.info(f"Evaluating model: {model_name}")

            # Use the tuned model if available
            model = self.tuned_models.get(model_name, self.models[model_name])

            # Perform cross-validation
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X_train, y_train, cv=cv_strategy, scoring="accuracy", n_jobs=-1
            )
            mean_score = scores.mean()
            std_score = scores.std()

            self.logger.info(
                f"{model_name} - Mean Accuracy: {mean_score:.4f}, Std: {std_score:.4f}"
            )

            # Check if this model performs better than the current best
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = model_name

        self.logger.info(
            f"Best model selected: {best_model_name} with cross-validated accuracy: {best_score:.4f}"
        )
        return best_model_name, best_score

    def compare_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
    ) -> pd.DataFrame:
        """
        Compares multiple models based on cross-validation accuracy scores.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            cv (int): Number of cross-validation folds.

        Returns:
            pd.DataFrame: DataFrame containing models and their scores.
        """
        self.logger.info("Comparing models...")
        results = []
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for model_name in self.models:
            self.logger.info(f"Evaluating model: {model_name}")
            model = self.tuned_models.get(model_name, self.models[model_name])
            scores = cross_val_score(
                model, X_train, y_train, cv=cv_strategy, scoring="accuracy", n_jobs=-1
            )
            results.append(
                {
                    "Model": model_name,
                    "Mean Accuracy": scores.mean(),
                    "Std Accuracy": scores.std(),
                }
            )
        results_df = pd.DataFrame(results)
        results_df.sort_values(by="Mean Accuracy", ascending=False, inplace=True)
        self.logger.info("Model comparison results:\n" + results_df.to_string(index=False))
        return results_df

    def plot_confusion_matrix(
        self, conf_matrix: np.ndarray, class_names: List[str]
    ) -> None:
        """
        Plots the confusion matrix.

        Args:
            conf_matrix (np.ndarray): The confusion matrix.
            class_names (List[str]): List of class names.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_roc_curve(
        self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """
        Plots the ROC curve for the given model.

        Args:
            model (BaseEstimator): The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing target.
        """
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                # Binary classification
                probabilities = y_proba[:, 1]
            else:
                # Multi-class classification
                self.logger.warning("ROC curve plotting is currently only implemented for binary classification.")
                return
        elif hasattr(model, "decision_function"):
            probabilities = model.decision_function(X_test)
        else:
            self.logger.warning("Model does not have predict_proba or decision_function.")
            return

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="blue",
            label=f"ROC Curve (AUC = {roc_auc_score(y_test, probabilities):.2f})",
        )
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    def cross_validate_model(
        self, model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> Dict[str, float]:
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
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(
            model, X, y, cv=cv_strategy, scoring="accuracy", n_jobs=-1
        )
        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
        }

    def plot_feature_importance(
        self, model: BaseEstimator, feature_names: List[str]
    ) -> None:
        """
        Plots feature importance for tree-based models or models that support it.

        Args:
            model (BaseEstimator): The trained model.
            feature_names (List[str]): List of feature names.
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importance")
            plt.bar(
                range(len(importances)),
                importances[indices],
                align="center",
                color="skyblue",
            )
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=90,
            )
            plt.tight_layout()
            plt.show()
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importance")
            plt.bar(
                range(len(importances)),
                importances[indices],
                align="center",
                color="skyblue",
            )
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=90,
            )
            plt.tight_layout()
            plt.show()
        else:
            self.logger.warning(f"Model {model.__class__.__name__} does not support feature importances.")

    def plot_learning_curve(
        self,
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> None:
        """
        Plots the learning curve of the model.

        Args:
            model (BaseEstimator): The model to plot learning curve for.
            X_train (pd.DataFrame): Feature matrix.
            y_train (pd.Series): Target variable.
            cv (int): Number of cross-validation folds.
            scoring (str): Scoring metric.

        """
        self.logger.info("Plotting learning curve...")
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            random_state=self.random_state,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes,
            train_scores_mean,
            "o-",
            color="r",
            label="Training score",
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel(scoring.capitalize())
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def save_model(self, model: BaseEstimator, filename: str) -> None:
        """
        Saves the trained model to disk.

        Args:
            model (BaseEstimator): The trained model.
            filename (str): The path and filename to save the model.
        """
        joblib.dump(model, filename)
        self.logger.info(f"Model saved to {filename}")

    def load_model(self, filename: str) -> BaseEstimator:
        """
        Loads a trained model from disk.

        Args:
            filename (str): The path and filename to load the model from.

        Returns:
            BaseEstimator: The loaded model.
        """
        model = joblib.load(filename)
        self.logger.info(f"Model loaded from {filename}")
        return model
