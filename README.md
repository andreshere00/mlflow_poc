# MLflow Overview

[**MLflow**](https://mlflow.org/) is an open-source platform designed to manage the end-to-end machine learning (ML) lifecycle—from development and training to deployment and artifact management. It is framework-agnostic (works with scikit-learn, TensorFlow, PyTorch, etc.) and environment-agnostic, making it easier to collaborate and ensure reproducibility in data science projects.

---

## Key Features of MLflow

- **Experiment Tracking:** Log, compare, and visualize parameters, metrics, and results from each ML run.
- **Model Management:** Store, version, review, and deploy ML models centrally.
- **Artifact Storage:** Save generated files such as models, images, logs, and more.
- **Reproducibility:** Rebuild experiments and models by recording environments and dependencies.
- **Web User Interface:** Provides intuitive dashboards for visualizing and comparing runs.
- **Multi-platform Support:** Works locally, on servers, and in the cloud.
- **CI/CD Integration:** Supports continuous deployment of ML models into production.

---

## Main components

1. **MLflow Tracking**
    - System to log and query ML experiments.
    - Stores parameters, metrics, artifacts, and results for each run.
    - Offers a web UI to visualize and compare experiments.

2. **MLflow Projects**
    - Standardizes how ML projects are packaged.
    - Defines the runtime environment (conda, docker, etc.) and training commands.
    - Makes experiments portable and reproducible.

3. **MLflow Models**
    - Provides a standard format for packaging ML models.
    - Supports storing, versioning, and deploying models on various platforms (local, REST API, AWS SageMaker, etc.).
    - Includes support for multiple "flavors" (scikit-learn, TensorFlow, PyTorch, etc.).

4. **MLflow Model Registry**
    - Centralized model store for versioning and lifecycle management.
    - Allows stage transitions (*staging*, *production*, *archived*).
    - Manages approvals, annotations, and model reviews for teams.