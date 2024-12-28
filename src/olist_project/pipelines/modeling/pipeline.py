"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    build_abt,
    train_oot_split
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_abt,
            inputs=[
                "modeling_spine",
                "modeling_feature_table",
                "params:audience_building.id_col",
                "params:audience_building.cohort_col",
            ],
            outputs="abt",
            name="build_abt_node",
        ),
        node(
            func=train_oot_split,
            inputs=[
                "abt",
                "params:modeling.start_cohort",
                "params:modeling.split_cohort",
                "params:modeling.end_cohort",
                "params:audience_building.id_col",
                "params:audience_building.cohort_col",
                "params:modeling.target",
            ],
            outputs=[
                "X_train",
                "y_train",
                "id_model_train",
                "X_test_oot",
                "y_test_oot",
                "id_model_test_oot",
            ],
            name="train_oot_split_node",
            tags=["modeling-without-abt-update"]
        ),
    ])
