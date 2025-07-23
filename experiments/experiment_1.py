"""
This script train a basic ElasticNet model and produce the results. The traceability is conducted
using the logging module, and it shows how can you simply obtain the traces for the training execution.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ElasticNet on wine quality data.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Regularization strength")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="L1 ratio for ElasticNet")
    return parser.parse_args()


def eval_metrics(actual: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("/tmp/app.log", mode="w"),
        logging.StreamHandler(),
    ],
)

FILE = "red-wine-quality.csv"
DATA_FOLDER = "data"
FILE_PATH = Path(DATA_FOLDER) / Path(FILE)
warnings.filterwarnings("ignore")

def main():
    args = get_args()

    logger.info(f"File path: {FILE_PATH}")
    logger.info(f"Performing the training method with the following arguments: {args}")

    # Read data
    data = pd.read_csv(FILE_PATH)
    data.to_csv(FILE_PATH, index=False)

    # Split train/test
    train, test = train_test_split(data, test_size=0.25, random_state=40)

    # Prepare features and targets
    X_train = train.drop(columns="quality")
    y_train = train["quality"]
    X_test = test.drop(columns="quality")
    y_test = test["quality"]

    logger.info("Test and train sets created. Objective variable defined.")

    # Model
    logger.info("Instantiating model...")
    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)

    logger.info("Fitting the model...")
    model.fit(X_train, y_train)

    logger.info("Predicting results...")
    predictions = model.predict(X_test)

    # Metrics
    logger.info("Evaluating model...")
    rmse, mae, r2 = eval_metrics(y_test, predictions)

    print(f"ElasticNet model (alpha={args.alpha}, l1_ratio={args.l1_ratio}):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()
