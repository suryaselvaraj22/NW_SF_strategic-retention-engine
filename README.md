# Strategic Retention & Churn Prediction Engine

![Databricks](https://img.shields.io/badge/Databricks-Serverless-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Supervised Learning](https://img.shields.io/badge/Machine_Learning-XGBoost%20%2F%20GBT-0194E2?style=for-the-badge)

## Executive Summary
This project implements an end-to-end Machine Learning pipeline to predict **Customer Churn Probability** for an enterprise insurance/call-center environment.

By leveraging Gradient Boosted Trees (XGBoost architecture) within a Databricks Unity Catalog environment, this solution enables:
* **Reduced "Cost-to-Serve":** Shifting from reactive damage control to proactive, automated retention workflows.
* **Maximized Customer Lifetime Value (LTV):** Identifying at-risk accounts before they churn and deploying targeted interventions (e.g., grace periods, human advisory calls).
* **Quantifiable Friction:** Translating operational pain points (support calls, payment delays) into direct P&L impact.

## The Tech Stack
* **Core Logic:** Python, PySpark (Spark MLlib)
* **Platform:** Databricks Serverless (Unity Catalog)
* **Modeling & Evaluation:** `GBTClassifier`, `VectorAssembler`, `BinaryClassificationEvaluator`
* **Data Storage:** Managed Delta Tables

## Key Results & Business Impact
The Gradient Boosted Tree classification model achieved near-perfect predictive accuracy, successfully untangling non-linear behavioral relationships. The model recorded an **AUC-ROC Score of 0.9976** and an **Accuracy of 97.81%** on unseen testing data.

### Extracted Business Intelligence (Feature Importance)
By mathematically extracting the tree-node weights, the model successfully isolated the top leading indicators of churn without human bias:
* **Support Call Frequency (0.84 Weight):** The overwhelming primary driver of churn. Customers calling 4+ times in 6 months require immediate tier-2 intervention.
* **Tenure (0.07 Weight):** First-year policyholders exhibit exponentially higher risk compared to 10-year veterans, validating the need for specialized "Year One" onboarding drip campaigns.
* **Payment Delays (0.07 Weight):** Financial friction serves as a critical early warning system for policy cancellation.

## Solution Architecture

This repository is modularized into a 3-stage enterprise pipeline:

### `01_churn_data_simulation.py`
Engineered a highly realistic synthetic dataset of 100,000 policyholders. Purposefully embedded hidden mathematical rules tying support call volume and payment delays to high churn probability, simulating real-world operational friction.

### `02_feature_engineering.py`
Prepared the data for tree-based modeling utilizing `VectorAssembler`. Implemented a reproducible **80/20 Train/Test Split** to ensure the model could be rigorously evaluated against unseen data, avoiding data leakage. *(Note: Feature scaling was purposefully omitted as tree-based models are scale-invariant).*

### `03_churn_modeling.py`
Trained the `GBTClassifier` on the 80% training set and evaluated its predictive power on the 20% holdout set. Generated robust evaluation metrics (AUC-ROC, Accuracy) and extracted a clean, human-readable Feature Importance matrix to drive strategic business action.

## How to Run This Project
1. Clone this repository to your local machine using VS Code.
2. Link the repository to your Databricks Workspace via Git integration.
3. Ensure you are running a Databricks cluster with Unity Catalog enabled.
4. Run the scripts sequentially (`01` through `03`).