"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import scoring


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scoring,
            inputs=["custom_lgbm_model",
                    "scoring.feature_table",
                    "params:audience_building.id_col",
                    "params:audience_building.cohort_col"],
            outputs="output",
            name="scoring_node"
        ),
    ], tags=["scoring", "scoring-without-preprocess"])
