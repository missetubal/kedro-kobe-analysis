import pandas as pd
import mlflow
from sklearn.metrics import log_loss, f1_score
from pathlib import Path
from utils import load_best_model_from_last_run


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("model_inference_pipeline")

    model_features = [
        "lat",
        "lon",
        "minutes_remaining",
        "period",
        "playoffs",
        "shot_distance",
        "shot_made_flag",
    ]

    df_prod = pd.read_parquet("data/01_raw/dataset_kobe_prod.parquet")
    df_prod = df_prod[model_features]

    if "shot_made_flag" in df_prod.columns:
        y_true = df_prod["shot_made_flag"]
        X = df_prod.drop(columns=["shot_made_flag"])
    else:
        y_true = None
        X = df_prod.copy()

    model, run_id = load_best_model_from_last_run()

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    df_prod["pred"] = y_pred
    if y_proba is not None:
        df_prod["proba"] = y_proba

    with mlflow.start_run(run_name="PipelineAplicacao"):
        if y_true is not None and not y_true.isnull().any():
            loss = log_loss(y_true, y_proba) if y_proba is not None else None
            f1 = f1_score(y_true, y_pred, average="macro")
            if loss:
                mlflow.log_metric("log_loss", loss)
            mlflow.log_metric("f1_score", f1)

        output_path = Path("data/06_models/app_prediction_pipeline.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_prod.to_parquet(output_path, index=False)

        mlflow.log_artifact(str(output_path))

        print("Predictions saved!")


if __name__ == "__main__":
    main()
