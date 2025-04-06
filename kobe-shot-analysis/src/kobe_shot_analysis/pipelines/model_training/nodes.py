from pycaret.classification import setup, predict_model, create_model, save_model
from sklearn.metrics import log_loss, f1_score
import mlflow
import mlflow.sklearn
import pandas as pd


def train_and_evaluate_models(base_train: pd.DataFrame, base_test: pd.DataFrame):

    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("model_training_pipeline")

    with mlflow.start_run(run_name="Treinamento") as run:
        clf = setup(
            data=base_train,
            target="shot_made_flag",
            session_id=42,
            log_experiment=True,
            experiment_name="Treinamento",
            verbose=False,
        )

        models = {"Logistic Regression": "lr", "DecisionTreeClassifier": "dt"}

        results = {}

        for model_name, model_cod in models.items():
            print(f"Training model: {model_name}")
            model = create_model(model_cod)

            predictions = predict_model(model, data=base_test)
            y_true = base_test["shot_made_flag"]

            try:
                y_proba = model.predict_proba(
                    base_test.drop(columns=["shot_made_flag"])
                )[:, 1]
            except:
                print(f"Error predicting model: {model_name}")
                y_proba = (
                    predictions["Score"] if "Score" in predictions.columns else None
                )

            if y_proba is not None:
                loss = log_loss(y_true, y_proba)
                mlflow.log_metric("log_loss", loss)
                results[model_name] = {"log_loss": loss}
            else:
                print(f"Score is missing - {model_name}")

            if model_cod == "dt":
                y_pred = (
                    predictions["Label"] if "Label" in predictions.columns else None
                )
                if y_pred is not None:
                    f1 = f1_score(y_true, y_pred, average="macro")
                    mlflow.log_metric("f1_score", f1)
                    results[model_name]["f1_score"] = f1

            mlflow.log_param("model", model_name)
            mlflow.log_param("base_train_len", len(base_train))
            mlflow.log_param("base_test_len", len(base_test))

            mlflow.sklearn.log_model(model, model_name)
            model_uri = f"runs:/{run.info.run_id}/{model_name}"
            registered_model_name = model_name.replace(" ", "_")

            mlflow.register_model(model_uri=model_uri, name=registered_model_name)

            print(f"Model {model_name} trained and logged")

            mlflow.end_run()

        best_model_name = min(results, key=lambda k: results[k]["log_loss"])
        print(
            f"Best model: {best_model_name} with log_loss = {results[best_model_name]['log_loss']:.4f}"
        )

        # Recria o melhor modelo e salva com nome fixo
        best_model = create_model(models[best_model_name])
        mlflow.sklearn.log_model(best_model, best_model_name)
        model_uri = f"runs:/models/{model_name}"
        registered_model_name = model_name.replace(" ", "_")

        print(f"Best model '{best_model_name}' saved as 'best_model'")

        return best_model_name
