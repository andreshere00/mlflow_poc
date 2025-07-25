import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
import mlflow

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("red-wine-quality.csv")
    #os.mkdir("data/")
    data.to_csv("data/red-wine-quality.csv", index=False)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())

# <!---- First Experiment Elastic Net ----!>

    print("First Experiment Elastic Net")
    exp = mlflow.set_experiment(experiment_name="exp_multi_EL")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
   # print("Artifact Location: {}".format(exp.artifact_location))
   # print("Tags: {}".format(exp.tags))
   # print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
   # print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    #log parameters
    params ={
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)
    #log metrics
    metrics = {
        "rmse":rmse,
        "r2":r2,
        "mae":mae
    }
    mlflow.log_metrics(metrics)
    #log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )

    mlflow.end_run()

## <!---- Second run ----!>

    mlflow.start_run(run_name="run2.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = ElasticNet(alpha=0.9, l1_ratio=0.9, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(0.9, 0.9))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": 0.9,
        "l1_ratio": 0.9
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()
    ########### Third run ######################

    mlflow.start_run(run_name="run3.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = ElasticNet(alpha=0.4, l1_ratio=0.4, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(0.4, 0.4))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": 0.4,
        "l1_ratio": 0.4
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

## <!---- Second Experiment Ridge ----!>

    print("Second Experiment Ridge")
    exp = mlflow.set_experiment(experiment_name="exp_multi_Ridge")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    # print("Artifact Location: {}".format(exp.artifact_location))
    # print("Tags: {}".format(exp.tags))
    # print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    # print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = Ridge(alpha=alpha,  random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge (alpha={:f}".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": alpha
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

## <!---- Second Run ----!>

    mlflow.start_run(run_name="run2.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = Ridge(alpha=0.9, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge (alpha={:f}".format(0.9))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": 0.9
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

## <!---- Third Run ----!>

    mlflow.start_run(run_name="run3.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = Ridge(alpha=0.4, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge (alpha={:f}".format(0.4))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": 0.4
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

## <!---- Third Experiment Ridge ----!>

    print("Third Experiment Ridge")
    exp = mlflow.set_experiment(experiment_name="exp_multi_Lasso")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    # print("Artifact Location: {}".format(exp.artifact_location))
    # print("Tags: {}".format(exp.tags))
    # print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    # print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = Lasso(alpha=alpha,  random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge (alpha={:f}".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": alpha
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

## <!---- Second run ----!>

    mlflow.start_run(run_name="run2.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = Lasso(alpha=0.9, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Ridge (alpha={:f}".format(0.9))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": 0.9
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

## <!---- Third Run ----!>

    mlflow.start_run(run_name="run3.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    lr = Lasso(alpha=0.4, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Lasso (alpha={:f}".format(0.4))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    # log parameters
    params = {
        "alpha": 0.4
    }
    mlflow.log_params(params)
    # log metrics
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }
    mlflow.log_metrics(metrics)
    # log model
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Recent Active run id is {}".format(run.info.run_id))
    print("Recent Active run name is {}".format(run.info.run_name))