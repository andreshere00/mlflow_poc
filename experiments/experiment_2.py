"""
This experiment trains and evaluates an ElasticNet regression model to predict the quality of red wine
using the 'red-wine-quality.csv' dataset. The script includes the full machine learning workflow:
reading and splitting the data, training the model with configurable hyperparameters (alpha and l1_ratio),
computing regression metrics (RMSE, MAE, R2), and tracking all relevant parameters, metrics, and the trained
model artifact using MLflow for experiment traceability and reproducibility. The script is intended to
demonstrate best practices in model training, evaluation, and experiment tracking for regression tasks.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow

# --- Logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("/tmp/app.log", mode="w"),
        logging.StreamHandler(),
    ],
)
warnings.filterwarnings("ignore")

# --- Constants ---
FILE = "red-wine-quality.csv"
DATA_FOLDER = "data"
FILE_PATH = Path(DATA_FOLDER) / Path(FILE)

# --- Argument parsing ---
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ElasticNet on wine quality data.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Regularization strength")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="L1 ratio for ElasticNet")
    return parser.parse_args()

# --- Data loading ---
def load_data(file_path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    return data

# --- Data splitting ---
def split_data(
    data: pd.DataFrame, 
    test_size: float = 0.25, 
    random_state: int = 40
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    logger.info("Data split into train and test sets.")
    return train, test

# --- Feature/target separation ---
def get_features_targets(
    df: pd.DataFrame, 
    target_col: str = "quality"
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

# --- Model training ---
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    alpha: float, 
    l1_ratio: float
) -> Any:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model trained.")
    return model

# --- Evaluation ---
def eval_metrics(actual: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# --- MLflow logging ---
def log_to_mlflow(
    model, 
    alpha: float, 
    l1_ratio: float, 
    rmse: float, 
    mae: float, 
    r2: float, 
    X_sample: pd.DataFrame,
    experiment_name: str = "experiment_2"
):
    exp = mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, name="regression_model", input_example=X_sample)
        logger.info("Metrics and model logged to MLflow.")

# --- Main pipeline ---
def main():
    args = get_args()

    logger.info(f"File path: {FILE_PATH}")
    logger.info(f"Training ElasticNet with arguments: {args}")

    # Data
    data = load_data(FILE_PATH)
    train_df, test_df = split_data(data)
    X_train, y_train = get_features_targets(train_df)
    X_test, y_test = get_features_targets(test_df)

    # Model
    model = train_model(X_train, y_train, args.alpha, args.l1_ratio)
    predictions = model.predict(X_test)

    # Metrics
    rmse, mae, r2 = eval_metrics(y_test, predictions)
    logger.info(f"Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # MLflow logging
    log_to_mlflow(
        model=model,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        rmse=rmse,
        mae=mae,
        r2=r2,
        X_sample=X_test.iloc[[0]],
    )

if __name__ == "__main__":
    main()
