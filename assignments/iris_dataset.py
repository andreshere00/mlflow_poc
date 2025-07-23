from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import mlflow

# Initializing dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Save input dataset as CSV (log as artifact)
input_data_df = pd.DataFrame(X, columns=iris.feature_names)
input_data_df['target'] = y
input_data_df.to_csv("data/iris_dataset.csv", index=False)

# Cross-Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save train/test splits as CSV
train_df = pd.DataFrame(X_train, columns=iris.feature_names)
train_df['target'] = y_train
train_df.to_csv("data/iris_train.csv", index=False)

test_df = pd.DataFrame(X_test, columns=iris.feature_names)
test_df['target'] = y_test
test_df.to_csv("data/iris_test.csv", index=False)

input_example = pd.DataFrame(X_train[:1], columns=iris.feature_names)

# Modelling
params = {
    "n_neighbors": 3,
    "algorithm": "auto",
    "leaf_size": 10,
    "metric": "euclidean"
}

classifier = neighbors.KNeighborsClassifier(**params)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')

# MLflow Tracking
mlflow.set_tracking_uri(uri="./mlruns")
mlflow.set_experiment("assignment_1")

with mlflow.start_run():
    # 1. Trace the parameters
    mlflow.log_params(params)
    
    # 2. Trace the metrics
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": pre,
        "recall": rec
    })

    # 3. Trace the model
    mlflow.sklearn.log_model(
        classifier, 
        "model",
        input_example=input_example
    )
    
    # 4. Log datasets as artifacts
    mlflow.log_artifact("data/iris_dataset.csv")
    mlflow.log_artifact("data/iris_train.csv")
    mlflow.log_artifact("data/iris_test.csv")
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {rec:.4f}")
