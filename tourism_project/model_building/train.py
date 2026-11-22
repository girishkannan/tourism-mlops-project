"""Model training with hyperparameter tuning and MLflow experiment tracking."""
import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-wellness-package")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")), categorical_features),
)

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, eval_metric="logloss"
)

param_grid = {
    "xgbclassifier__n_estimators": [75, 150],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [0.5, 1.0],
}

pipeline = make_pipeline(preprocessor, xgb_model)
CLASSIFICATION_THRESHOLD = 0.45

with mlflow.start_run(run_name="xgboost-grid-search"):
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="recall", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # Log every tuning combination as a nested run
    results = grid.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_recall", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_recall", results["std_test_score"][i])

    best_model = grid.best_estimator_
    mlflow.log_params(grid.best_params_)

    train_pred = (best_model.predict_proba(Xtrain)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
    test_pred = (best_model.predict_proba(Xtest)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)

    train_report = classification_report(ytrain, train_pred, output_dict=True)
    test_report = classification_report(ytest, test_pred, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    print("Best params:", grid.best_params_)
    print("\nTest classification report:")
    print(classification_report(ytest, test_pred))

    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
