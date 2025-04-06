import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split


def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("data_processing_pipeline")

    with mlflow.start_run(run_name="PrepaaracaoDados"):
        selected_columns = [
            "lat",
            "lon",
            "minutes_remaining",
            "period",
            "playoffs",
            "shot_distance",
            "shot_made_flag",
        ]

        data = raw_data[selected_columns]

        mlflow.log_metric("linhas", len(data))
        mlflow.log_metric("colunas", len(data.columns))

        original_shape = data.shape
        data = data.dropna()
        cleaned_shape = data.shape

        mlflow.log_metric("linhas_nao_nulas", cleaned_shape[0])
        mlflow.log_metric("colunas_nao_nulas", cleaned_shape[1])
        mlflow.log_metric("linhas_removidas", original_shape[0] - cleaned_shape[0])
        mlflow.log_metric("colunas_removidas", original_shape[1] - cleaned_shape[1])

        return data


def split_data(data: pd.DataFrame, test_size: float = 0.2) -> dict:
    if "shot_made_flag" not in data.columns:
        raise ValueError("A coluna 'shot_made_flag' não está presente no dataset.")

    X = data.drop(columns=["shot_made_flag"])
    y = data["shot_made_flag"]

    with mlflow.start_run(run_name="SplitData"):
        mlflow.log_param("test_size", test_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    mlflow.log_metric("train_size", train_df.shape[0])
    mlflow.log_metric("test_size", test_df.shape[0])

    return {"base_train": train_df, "base_test": test_df}
