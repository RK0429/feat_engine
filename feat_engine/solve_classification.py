import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna  # Added for Bayesian optimization

from typing import Any, Dict, List, Optional, Tuple, Callable

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,  # Added for stacking
    BaggingClassifier,
    AdaBoostClassifier
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
                probability=True, random_state=self.random_state, max_iter=10000
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=self.random_state),
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
            "Stacking Classifier": StackingClassifier(  # Added StackingClassifier
                estimators=[
                    ("rf", RandomForestClassifier(random_state=self.random_state)),
                    ("svc", SVC(probability=True, random_state=self.random_state)),
                ],
                final_estimator=LogisticRegression(random_state=self.random_state),
                passthrough=True,
                cv=5,
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
                "penalty": ["l1", "l2", "elasticnet", None],
                "l1_ratio": [0.0, 0.5, 1.0],  # Only used if penalty='elasticnet'
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200, 500],
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2", 0.1, 0.5, 1.0],
                "bootstrap": [True, False],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.001, 0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10, 15],
                "subsample": [0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
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
                "metric": ["minkowski", "euclidean", "manhattan"],
            },
            "Decision Tree": {
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
                "max_features": ["auto", "sqrt", "log2", 0.1, 0.5, 1.0],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10, 15],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2, 0.3],
                "reg_alpha": [0, 0.01, 0.1, 1],
                "reg_lambda": [1, 1.5, 2, 3],
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
                "num_leaves": [31, 50, 100, 150],
                "max_depth": [-1, 10, 20, 30],
                "min_child_samples": [20, 30, 40, 50],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1],
                "reg_lambda": [0, 0.01, 0.1, 1],
            },
            "CatBoost": {
                "iterations": [100, 200, 500],
                "learning_rate": [0.001, 0.01, 0.1, 0.2],
                "depth": [3, 5, 7, 10],
                "l2_leaf_reg": [1, 3, 5, 7, 9],
                "border_count": [32, 64, 128, 256],
                "bagging_temperature": [0, 1, 3, 5],
            },
            "Naive Bayes": {
                "var_smoothing": np.logspace(0, -9, num=100),
            },
            "Voting Classifier": {
                "voting": ["soft", "hard"],
                "weights": [
                    [1, 1, 1],  # Equal weights for LogisticRegression, RandomForest, SVC
                    [2, 1, 1],  # Favor LogisticRegression more
                    [1, 2, 1],  # Favor RandomForest more
                    [1, 1, 2],  # Favor SVC more
                ],
            },
            "Stacking Classifier": {  # Added parameter grid for StackingClassifier
                "final_estimator__C": [0.01, 0.1, 1, 10],
                "final_estimator__solver": ["lbfgs", "liblinear"],
                "final_estimator__penalty": ["l2"],
            },
        }

    def _default_bayesian_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """
        Provides fine-grained default search spaces for Bayesian optimization for each model.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of parameter distributions suitable for Optuna.
        """
        return {
            "Logistic Regression": {
                "C": optuna.distributions.FloatDistribution(low=1e-4, high=1e4, log=True),  # Updated
                "solver": optuna.distributions.CategoricalDistribution(choices=["liblinear", "lbfgs", "saga", "newton-cg"]),
                "penalty": optuna.distributions.CategoricalDistribution(choices=["l1", "l2", "elasticnet", None]),
                "l1_ratio": optuna.distributions.FloatDistribution(low=0.0, high=1.0, log=False),  # Updated
            },
            "Random Forest": {
                "n_estimators": optuna.distributions.IntDistribution(low=10, high=1000, log=False),  # Updated
                "max_depth": optuna.distributions.IntDistribution(low=1, high=100, log=False),  # Updated
                "min_samples_split": optuna.distributions.IntDistribution(low=2, high=20, log=False),  # Updated
                "min_samples_leaf": optuna.distributions.IntDistribution(low=1, high=20, log=False),  # Updated
                "max_features": optuna.distributions.CategoricalDistribution(choices=["auto", "sqrt", "log2"]),
            },
            "Gradient Boosting": {
                "n_estimators": optuna.distributions.IntDistribution(low=50, high=1000, log=False),  # Updated
                "learning_rate": optuna.distributions.FloatDistribution(low=1e-4, high=1.0, log=True),  # Updated
                "max_depth": optuna.distributions.IntDistribution(low=3, high=20, log=False),  # Updated
                "subsample": optuna.distributions.FloatDistribution(low=0.5, high=1.0, log=False),  # Updated
                "min_samples_split": optuna.distributions.IntDistribution(low=2, high=20, log=False),  # Updated
                "min_samples_leaf": optuna.distributions.IntDistribution(low=1, high=20, log=False),  # Updated
            },
            "Support Vector Machine": {
                "C": optuna.distributions.FloatDistribution(low=1e-4, high=1e4, log=True),  # Updated
                "kernel": optuna.distributions.CategoricalDistribution(choices=["linear", "rbf", "poly", "sigmoid"]),
                "gamma": optuna.distributions.CategoricalDistribution(choices=["scale", "auto"]),
            },
            "K-Nearest Neighbors": {
                "n_neighbors": optuna.distributions.IntDistribution(low=1, high=30, log=False),  # Updated
                "weights": optuna.distributions.CategoricalDistribution(choices=["uniform", "distance"]),
                "p": optuna.distributions.IntDistribution(low=1, high=2, log=False),  # Updated
                "metric": optuna.distributions.CategoricalDistribution(choices=["minkowski", "euclidean", "manhattan"]),
            },
            "Decision Tree": {
                "max_depth": optuna.distributions.IntDistribution(low=1, high=100, log=False),  # Updated
                "min_samples_split": optuna.distributions.IntDistribution(low=2, high=20, log=False),  # Updated
                "min_samples_leaf": optuna.distributions.IntDistribution(low=1, high=20, log=False),  # Updated
                "criterion": optuna.distributions.CategoricalDistribution(choices=["gini", "entropy"]),
                "max_features": optuna.distributions.CategoricalDistribution(choices=["auto", "sqrt", "log2"]),
            },
            "XGBoost": {
                "n_estimators": optuna.distributions.IntDistribution(low=50, high=1000, log=False),  # Updated
                "learning_rate": optuna.distributions.FloatDistribution(low=1e-4, high=1.0, log=True),  # Updated
                "max_depth": optuna.distributions.IntDistribution(low=1, high=20, log=False),  # Updated
                "subsample": optuna.distributions.FloatDistribution(low=0.5, high=1.0, log=False),  # Updated
                "colsample_bytree": optuna.distributions.FloatDistribution(low=0.5, high=1.0, log=False),  # Updated
                "gamma": optuna.distributions.FloatDistribution(low=0.0, high=5.0, log=False),  # Updated
                "reg_alpha": optuna.distributions.FloatDistribution(low=0.0, high=5.0, log=False),  # Updated
                "reg_lambda": optuna.distributions.FloatDistribution(low=1.0, high=5.0, log=False),  # Updated
            },
            "LightGBM": {
                "n_estimators": optuna.distributions.IntDistribution(low=50, high=1000, log=False),  # Updated
                "learning_rate": optuna.distributions.FloatDistribution(low=1e-4, high=1.0, log=True),  # Updated
                "num_leaves": optuna.distributions.IntDistribution(low=20, high=150, log=False),  # Updated
                "max_depth": optuna.distributions.IntDistribution(low=1, high=100, log=False),  # Updated
                "min_child_samples": optuna.distributions.IntDistribution(low=10, high=100, log=False),  # Updated
                "subsample": optuna.distributions.FloatDistribution(low=0.5, high=1.0, log=False),  # Updated
                "colsample_bytree": optuna.distributions.FloatDistribution(low=0.5, high=1.0, log=False),  # Updated
                "reg_alpha": optuna.distributions.FloatDistribution(low=0.0, high=5.0, log=False),  # Updated
                "reg_lambda": optuna.distributions.FloatDistribution(low=0.0, high=5.0, log=False),  # Updated
            },
            "CatBoost": {
                "iterations": optuna.distributions.IntDistribution(low=100, high=1000, log=False),  # Updated
                "learning_rate": optuna.distributions.FloatDistribution(low=1e-4, high=1.0, log=True),  # Updated
                "depth": optuna.distributions.IntDistribution(low=3, high=16, log=False),  # Updated
                "l2_leaf_reg": optuna.distributions.IntDistribution(low=1, high=10, log=False),  # Updated
                "border_count": optuna.distributions.IntDistribution(low=32, high=256, log=False),  # Updated
                "bagging_temperature": optuna.distributions.FloatDistribution(low=0.0, high=5.0, log=False),  # Updated
            },
            "Naive Bayes": {
                "var_smoothing": optuna.distributions.FloatDistribution(low=1e-12, high=1e-6, log=True),  # Updated
            },
            "Voting Classifier": {
                "voting": optuna.distributions.CategoricalDistribution(choices=["soft", "hard"]),
                "weights": optuna.distributions.CategoricalDistribution(choices=[
                    (1, 1, 1),  # Equal weights for LogisticRegression, RandomForest, SVC
                    (2, 1, 1),  # Favor LogisticRegression more
                    (1, 2, 1),  # Favor RandomForest more
                    (1, 1, 2),  # Favor SVC more
                ]),
            },
            "Stacking Classifier": {  # Added parameter grid for StackingClassifier
                "final_estimator__C": optuna.distributions.FloatDistribution(low=1e-4, high=1e4, log=True),  # Updated
                "final_estimator__solver": optuna.distributions.CategoricalDistribution(choices=["lbfgs", "liblinear"]),
                "final_estimator__penalty": optuna.distributions.CategoricalDistribution(choices=["l2"]),
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
        param_distributions: Optional[Dict[str, Any]] = None,  # New argument for Bayesian
        cv: int = 5,
        search_type: str = "grid",
        n_iter: int = 50,
        scoring: str = "accuracy",
    ) -> None:
        """
        Performs hyperparameter tuning using GridSearchCV, RandomizedSearchCV, or Bayesian Optimization for one or all models and stores the best models.

        Args:
            model_name (str): The name of the model to tune. If 'all', tunes all models in self.models.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            param_grid (Optional[Dict[str, List[Any]]]): Parameter grid for hyperparameter tuning. If None, uses default.
            param_distributions (Optional[Dict[str, Any]]): Parameter distributions for Bayesian optimization.
            cv (int): Number of cross-validation folds.
            search_type (str): Type of search ('grid', 'random', or 'bayesian').
            n_iter (int): Number of iterations for RandomizedSearchCV or Bayesian Optimization.
            scoring (str): Scoring metric for evaluation.

        Returns:
            None: The best models are stored in self.tuned_models.
        """
        if model_name == "all":
            self.logger.info("Performing hyperparameter tuning for all models...")
            for name in self.models:
                self._tune_single_model(
                    name, X_train, y_train, param_grid, param_distributions, cv, search_type, n_iter, scoring
                )
        else:
            self._tune_single_model(
                model_name,
                X_train,
                y_train,
                param_grid,
                param_distributions,
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
        param_distributions: Optional[Dict[str, Any]],
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
            param_distributions (Optional[Dict[str, Any]]): Parameter distributions for Bayesian optimization.
            cv (int): Number of cross-validation folds.
            search_type (str): Type of search ('grid', 'random', or 'bayesian').
            n_iter (int): Number of iterations for RandomizedSearchCV or Bayesian Optimization.
            scoring (str): Scoring metric for evaluation.

        Returns:
            None: The best model is stored in self.tuned_models.
        """
        model = self.models[model_name]
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")

        # Use param_grid if provided, else use default
        if param_grid is None and search_type != "bayesian":
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
            search.fit(X_train, y_train.values.ravel())
            self.logger.info(
                f"Best parameters found for {model_name}: {search.best_params_}"
            )
            # Store the tuned model for future use
            self.tuned_models[model_name] = search.best_estimator_

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
            search.fit(X_train, y_train.values.ravel())
            self.logger.info(
                f"Best parameters found for {model_name}: {search.best_params_}"
            )
            # Store the tuned model for future use
            self.tuned_models[model_name] = search.best_estimator_

        elif search_type == "bayesian":  # Added Bayesian optimization
            self.logger.info(f"Using Bayesian Optimization for {model_name}...")
            # Use param_distributions if provided, else use default Bayesian search spaces
            if param_distributions is None:
                param_distributions = self._default_bayesian_search_spaces().get(model_name, {})
                if not param_distributions:
                    self.logger.warning(
                        f"No Bayesian parameter distribution available for {model_name}. Skipping tuning."
                    )
                    return

            study = optuna.create_study(direction="maximize" if 'accuracy' in scoring else "minimize")
            func = self._create_objective(model, param_distributions, X_train, y_train, cv, scoring)
            study.optimize(func, n_trials=n_iter, show_progress_bar=True)
            best_params = study.best_params
            self.logger.info(f"Best parameters found for {model_name}: {best_params}")
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            self.tuned_models[model_name] = model

        else:
            raise ValueError("search_type must be either 'grid', 'random', or 'bayesian'.")

    def _create_objective(
        self,
        model: BaseEstimator,
        param_distributions: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        cv: int,
        scoring: str,
    ) -> Callable[[Any], Any]:
        """
        Creates an objective function for Optuna to optimize model hyperparameters.

        Args:
            model (BaseEstimator): The machine learning model to be optimized.
            param_distributions (Dict[str, Any]): The distributions of hyperparameters to sample from.
            X (pd.DataFrame): Training data for features.
            y (pd.Series): Target labels for training data.
            cv (int): The number of cross-validation folds.
            scoring (str): The scoring metric to evaluate model performance.

        Returns:
            Callable[[Any], float]: The objective function to be minimized or maximized by Optuna.
        """

        def objective(trial: optuna.trial.Trial) -> Any:
            """
            The actual objective function used by Optuna to evaluate a set of hyperparameters.

            Args:
                trial (optuna.trial.Trial): A single trial instance that suggests hyperparameters.

            Returns:
                float: The mean cross-validated score for the suggested hyperparameter set.
            """
            params = {}
            for param, distribution in param_distributions.items():
                if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                    params[param] = trial.suggest_categorical(param, distribution.choices)
                elif isinstance(distribution, optuna.distributions.FloatDistribution):
                    if distribution.log:
                        params[param] = trial.suggest_loguniform(param, distribution.low, distribution.high)
                    else:
                        params[param] = trial.suggest_uniform(param, distribution.low, distribution.high)
                elif isinstance(distribution, optuna.distributions.IntDistribution):
                    params[param] = trial.suggest_int(param, distribution.low, distribution.high)
                else:
                    # Handle other distribution types if necessary
                    params[param] = trial.suggest_float(param, 0.0, 1.0)

            model.set_params(**params)

            # Use stratified k-fold cross-validation to evaluate the model
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            score = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1).mean()

            return score

        return objective

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
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Cross-validates the model using the specified number of folds and returns detailed metrics.

        Args:
            model (BaseEstimator): The model to cross-validate.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            cv (int): Number of cross-validation folds (default: 5).

        Returns:
            Dict[str, Any]: Cross-validation metrics including accuracy, precision, recall, f1, ROC AUC, etc.
        """
        self.logger.info("Cross-validating the provided model...")

        # Define stratified cross-validation
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        # Store metrics for each fold
        scoring_metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc", "neg_log_loss"]

        # Perform cross-validation for all required metrics
        scores = {}
        for metric in scoring_metrics:
            self.logger.info(f"Calculating {metric}...")
            try:
                score = cross_val_score(
                    model, X, y, cv=cv_strategy, scoring=metric, n_jobs=-1
                )
                scores[metric] = {
                    "mean": float(np.mean(score)),
                    "std": float(np.std(score)),
                }
            except ValueError as e:
                self.logger.warning(f"Skipping metric {metric} due to error: {e}")
                continue

        # Return detailed cross-validation results
        return scores

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
            shuffle=True,
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

    def model_merging(
        self,
        base_models: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = "stacking",  # New argument to select the ensemble method
        final_estimator: Optional[BaseEstimator] = None,
        passthrough: bool = False,
        cv: int = 5,
        n_estimators: int = 10  # For bagging and boosting
    ) -> BaseEstimator:
        """
        Creates an ensemble model by merging multiple base models using different ensemble techniques.
        Supports stacking, bagging, boosting, and voting.

        Args:
            base_models (List[str]): List of model names to be used as base models.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            method (str): The ensemble method to use ('stacking', 'bagging', 'boosting', or 'voting').
            final_estimator (Optional[BaseEstimator]): The final estimator to combine base models for stacking. Defaults to LogisticRegression.
            passthrough (bool): If True, pass the original features to the final estimator (only for stacking).
            cv (int): Number of cross-validation folds for stacking.
            n_estimators (int): Number of estimators for bagging or boosting.

        Returns:
            BaseEstimator: The ensemble model.
        """
        self.logger.info(f"Creating {method.capitalize()} Classifier for model merging...")

        # Collect base models
        estimators = []
        for model_name in base_models:
            if model_name in self.tuned_models:
                model = self.tuned_models[model_name]
                self.logger.info(f"Using tuned model: {model_name} for {method}.")
            else:
                model = self.models.get(model_name)
                if model is None:
                    self.logger.warning(f"Model {model_name} not found. Skipping.")
                    continue
                self.logger.info(f"Using default model: {model_name} for {method}.")
            estimators.append((model_name, model))

        if not estimators:
            self.logger.error(f"No valid base models provided for {method}.")
            raise ValueError(f"No valid base models provided for {method}.")

        # Define the final estimator for stacking
        if final_estimator is None and method == "stacking":
            final_estimator = LogisticRegression(random_state=self.random_state)

        # Ensemble Methods
        if method == "stacking":
            # Stacking Classifier
            ensemble_model = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                passthrough=passthrough,
                cv=cv,
                n_jobs=-1,
            )

        elif method == "bagging":
            # Bagging Classifier
            base_estimator = estimators[0][1] if len(estimators) == 1 else DecisionTreeClassifier(random_state=self.random_state)
            ensemble_model = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )

        elif method == "boosting":
            # Boosting Classifier (Gradient Boosting or AdaBoost)
            if "Gradient Boosting" in base_models:
                ensemble_model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                )
            else:
                # Default to AdaBoost if Gradient Boosting is not in base models
                ensemble_model = AdaBoostClassifier(
                    base_estimator=estimators[0][1] if len(estimators) == 1 else DecisionTreeClassifier(random_state=self.random_state),
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                )

        elif method == "voting":
            # Voting Classifier
            ensemble_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

        else:
            self.logger.error(f"Unknown ensemble method: {method}")
            raise ValueError(f"Unknown ensemble method: {method}")

        # Fit the ensemble model
        self.logger.info(f"Training {method.capitalize()} Classifier...")
        ensemble_model.fit(X_train, y_train)

        # Store the ensemble model in tuned_models for future use
        self.tuned_models[f"{method.capitalize()} Classifier"] = ensemble_model

        return ensemble_model
