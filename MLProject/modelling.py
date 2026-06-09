import mlflow
import mlflow.sklearn
import shutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
from pathlib import Path
import dagshub

dagshub.init(
    repo_owner="Sw4nhilde",
    repo_name="Membangun_model_Muhammad-Eka-Mandiri-Sujanto",
    mlflow=True
)

mlflow.set_experiment("customer_churn_modeling")
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    run_id_file = Path(__file__).resolve().parent / "latest_run_id.txt"
    run_id_file.write_text(run_id, encoding="utf-8")
    print(f"RUN_ID={run_id}")

    df = pd.read_csv(
        "customer_churn_processed.csv"
    )

    X = df.drop(
        "Churn",
        axis=1
    )

    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier()

    model.fit(
        X_train,
        y_train
    )

    score = model.score(
        X_test,
        y_test
    )

    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", score)
    mlflow.log_text(
        classification_report(y_test, y_pred),
        "classification_report.txt"
    )
    mlflow.log_text(
        str(confusion_matrix(y_test, y_pred)),
        "confusion_matrix.txt"
    )
    model_dir = Path(__file__).resolve().parent / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(model, path=str(model_dir))
    mlflow.log_artifacts(str(model_dir), artifact_path="model")

    print(score)