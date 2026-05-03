import argparse, mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth",    type=int, default=6)
parser.add_argument("--test-size",    type=float, default=0.2)
args = parser.parse_args()

with mlflow.start_run():
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=args.test_size, random_state=42
    )
    # Note: MLflow Projects auto-logs the -P params, no need to call log_params() here

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_te, y_pred),
        "roc_auc":  roc_auc_score(y_te, y_proba)
    })
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Project run complete! Accuracy: {accuracy_score(y_te, y_pred):.4f}")
