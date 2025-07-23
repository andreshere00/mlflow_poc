from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

import mlflow
import mlflow.sklearn

# Initializing dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Cross-Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {rec:.4f}")
