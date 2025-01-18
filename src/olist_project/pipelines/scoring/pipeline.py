"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import scoring


def create_pipeline(**kwargs) -> Pipeline:
    pipe_template = pipeline([
        node(
            func=scoring,
            inputs=["custom_lgbm_model",
                    "feature_table",
                    "params:audience_building.id_col",
                    "params:audience_building.cohort_col"],
            outputs="output",
            name="scoring_node"
        ),
    ], tags=["scoring-without-preprocess"])

    pipe_scoring = pipeline(
        pipe=pipe_template,
        namespace="scoring",
        inputs={
            "custom_lgbm_model": "modeling.custom_lgbm_model",
        },
        parameters={
            "params:audience_building.id_col",
            "params:audience_building.cohort_col"
        },
        tags=["scoring-without-preprocess"]
    )

    return pipe_scoring
