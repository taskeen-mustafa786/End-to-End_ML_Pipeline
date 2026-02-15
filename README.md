# End-to-End ML Pipeline
This repository demonstrates how to build a **reusable and production-ready machine learning pipeline** for predicting customer churn using the Scikit‑learn `Pipeline` API.

---

## 📋 Objective of the Task
The goal of this project is to construct an end-to-end solution that can ingest raw customer data, apply preprocessing, train classification models, perform hyperparameter tuning, and export a single artifact suitable for deployment. We focus on the IBM Telco Churn dataset and target the binary churn prediction problem.

## 🛠️ Methodology / Approach
1. **Data Acquisition & Cleaning**
   - Load the Telco Churn dataset from a public CSV URL.
   - Convert `TotalCharges` to numeric and drop rows with missing values.
   - Define features (`X`) and binary target (`y`).

2. **Preprocessing Pipeline**
   - Separate numeric and categorical columns automatically.
   - Numeric pipeline: median imputation + standard scaling.
   - Categorical pipeline: constant imputation + one-hot encoding (ignore unknowns).
   - Combine with `ColumnTransformer` so that future data flows through the same transformers.

3. **Model Training & Hyperparameter Search**
   - Wrap preprocessing and classifier into a single `Pipeline` object.
   - Evaluate both **Logistic Regression** and **Random Forest** classifiers.
   - Use `GridSearchCV` with 5‑fold cross‑validation and `f1` scoring for robust selection.

4. **Evaluation & Export**
   - Assess the best estimator on a hold‑out test set using classification report metrics.
   - Serialize the entire pipeline (preprocessor + model) with `joblib` for reuse in production environments (Flask/FastAPI, serverless, etc.).

## ✅ Key Results & Observations
- The grid search prints the best hyperparameters, e.g. a Random Forest with `n_estimators=200` and `max_depth=20` (example) often outperformed logistic regression on `f1` score.
- The final classification report on the test set provides precision, recall, and F1-score for the churn class.
- The pipeline exported as `telco_churn_pipeline.pkl` encapsulates all preprocessing logic and the trained model, ensuring consistency between training and inference.

> **Observation:** Using Scikit‑learn's `Pipeline` and `ColumnTransformer` simplifies deployment and reduces the risk of data leakage. Hyperparameter tuning within the pipeline keeps the workflow clean and repeatable.

## 📁 Dataset
**Telco Churn Dataset** – customer information from a telecommunications company; widely used for churn modeling tasks.

## 🎓 Skills Gained
- Constructing complex ML pipelines with scikit-learn.
- Performing hyperparameter tuning using `GridSearchCV`.
- Exporting pipelines for production use with `joblib`.
- Following production-readiness practices like modular preprocessing and model serialization.

---

Feel free to inspect `dh_int 02.ipynb` for the full implementation and experimentation steps.
