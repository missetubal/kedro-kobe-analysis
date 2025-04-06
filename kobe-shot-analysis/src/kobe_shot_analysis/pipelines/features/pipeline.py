from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_features,
                inputs="processed_prod_data",
                outputs="data_features_prod",
                name="create_features_node",
            )
        ]
    )
