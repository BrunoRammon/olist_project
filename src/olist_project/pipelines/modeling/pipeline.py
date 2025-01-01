"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    build_abt,
    train_oot_split,
    train_final_model,
    model_results
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_abt,
            inputs=[
                "modeling.spine",
                "modeling.feature_table",
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
                "params:modeling.features"
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
        node(
            func=train_final_model,
            inputs=[
                "X_train",
                "y_train",
                "params:modeling.ratings",
                "params:random_state",
                "params:modeling.hyperparameters",
                "params:modeling.nfolds_cv"
            ],
            outputs="custom_lgbm_model",
            name="train_lgbm_node",
            tags=["modeling-without-abt-update", "modeling-only-train"]
        ),
        node(
            func=model_results,
            inputs=[
                "custom_lgbm_model",
                "X_train",
                "y_train",
                "id_model_train",
                "X_test_oot",
                "y_test_oot",
                "id_model_test_oot",
            ],
            outputs=["results_train", "results_test_oot"],
            name="model_results_node",
            tags=["modeling-without-abt-update", "modeling-only-train"]
        ),
    ], tags=["modeling", "modeling-without-preprocess"])
