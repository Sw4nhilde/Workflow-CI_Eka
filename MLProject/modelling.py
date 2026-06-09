import mlflow
import mlflow.sklearn
import shutil
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
from pathlib import Path

# Check if running in GitHub Actions
if os.getenv('GITHUB_ACTIONS') == 'true':
    # Use local tracking in CI/CD
    mlflow.set_tracking_uri("file:./mlruns")
    print("Running in GitHub Actions - using local MLflow tracking")
else:
    # Use DagsHub tracking locally
    import dagshub
    dagshub.init(
        repo_owner="Sw4nhilde",
        repo_name="Membangun_model_Muhammad-Eka-Mandiri-Sujanto",
        mlflow=True
    )
    print("Running locally - using DagsHub tracking")

# Clear any existing runs
os.environ.pop('MLFLOW_RUN_ID', None)

# Start a new run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    run_id_file = Path(__file__).resolve().parent / "latest_run_id.txt"
    run_id_file.write_text(run_id, encoding="utf-8")
    print(f"RUN_ID={run_id}")

    # Load data
    df = pd.read_csv("customer_churn_processed.csv")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", score)
    mlflow.log_text(
        classification_report(y_test, y_pred),
        "classification_report.txt"
    )
    mlflow.log_text(
        str(confusion_matrix(y_test, y_pred)),
        "confusion_matrix.txt"
    )
    
    # Save and log model
    model_dir = Path(__file__).resolve().parent / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(model, path=str(model_dir))
    mlflow.log_artifacts(str(model_dir), artifact_path="model")

    print(f"Model accuracy: {score}")
