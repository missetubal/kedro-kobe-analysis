"""Project pipelines."""

from kobe_shot_analysis.pipelines import data_processing, model_training, features


def register_pipelines():
    return {
        "__default__": data_processing.create_pipeline(),
        "data_processing": data_processing.create_pipeline(),
        "model_training": model_training.create_pipeline(),
        "features": features.create_pipeline(),
    }
