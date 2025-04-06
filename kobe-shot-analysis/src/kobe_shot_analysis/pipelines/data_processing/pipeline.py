from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs="kobe_raw_data",
                outputs="processed_data",
                name="preprocess_data_node",
            ),
            node(
                func=split_data,
                inputs=dict(data="processed_data", test_size="params:test_size"),
                outputs=dict(base_train="base_train", base_test="base_test"),
                name="split_data_node",
            ),
            # prod
            node(
                func=preprocess_data,
                inputs="dataset_kobe_prod",
                outputs="processed_prod_data",
                name="preprocess_prod_data_node",
            ),
        ]
    )
