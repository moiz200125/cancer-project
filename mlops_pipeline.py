import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

import os

# Point to our running MLflow server ONLY if not in GitHub Actions
if not os.getenv("GITHUB_ACTIONS"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT  = "mlops-pipeline"
MODEL_NAME  = "ProductionClassifier"
ACCURACY_GATE = 0.999   # Impossible gate — testing that no model gets promoted

mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Some models (like Logistic Regression) need scaled data
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

candidates = [
    ("RandomForest",  RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42), False),
    ("GradBoost",     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), False),
    ("LogisticReg",   LogisticRegression(C=1.0, max_iter=1000, random_state=42), True),  # needs scaling
]

run_results = []

print(f"Starting Automated Pipeline for Experiment: {EXPERIMENT}...")
print("-" * 60)

for name, model, scaled in candidates:
    Xtr_use = X_tr_s if scaled else X_tr
    Xte_use = X_te_s if scaled else X_te

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_param("scaled", scaled)
        
        model.fit(Xtr_use, y_tr)
        
        acc  = accuracy_score(y_te, model.predict(Xte_use))
        auc  = roc_auc_score(y_te, model.predict_proba(Xte_use)[:, 1])
        
        mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        run_results.append({
            "name": name, 
            "run_id": run.info.run_id, 
            "accuracy": acc, 
            "roc_auc": auc
        })
        print(f"  {name:15s}  acc={acc:.4f}  auc={auc:.4f}")

# --- Automated Promotion Gate ---
# We pick the champion based on ROC AUC
best = max(run_results, key=lambda r: r["roc_auc"])
print("-" * 60)
print(f"Champion Candidate: {best['name']} (roc_auc={best['roc_auc']:.4f})")

if best["accuracy"] >= ACCURACY_GATE:
    # 1. Register the model
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    
    # 2. Promote to Production
    client.transition_model_version_stage(
        name=MODEL_NAME, 
        version=mv.version,
        stage="Production", 
        archive_existing_versions=True
    )
    print(f"\nPROMOTED {best['name']} v{mv.version} -> Production")
    print("This model is now live!")
else:
    print(f"\nGATE FAILED: accuracy {best['accuracy']:.4f} < {ACCURACY_GATE}")
    print("No model was promoted.")
