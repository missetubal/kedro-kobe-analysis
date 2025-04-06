import mlflow.sklearn


def load_best_model_from_last_run():
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name("model_training_pipeline")

    if experiment is None:
        raise Exception("Experiment not found")

    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id], order_by=["start_time DESC"]
    )

    if runs.empty:
        raise Exception("No runs found")

    last_run_id = runs.iloc[0]["run_id"]
    last_run_params_model = runs.iloc[0]["params.model"]

    model_uri = f"runs:/{last_run_id}/{last_run_params_model}"
    print(f"Loading best model from run_id: {last_run_id}")
    model = mlflow.sklearn.load_model(model_uri)

    return model, last_run_id
