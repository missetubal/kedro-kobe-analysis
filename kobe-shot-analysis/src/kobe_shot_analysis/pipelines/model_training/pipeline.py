from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_and_evaluate_models


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_and_evaluate_models,
                inputs=["base_train", "base_test"],
                outputs="best_model",
                name="train_model_node",
            )
        ]
    )
