"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    build_abt,
    train_oot_split,
    feature_selection,
    hyperparameters_tuning,
    ratings_optimization,
    train_final_model,
    model_results
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe_template = pipeline([
        node(
            func=build_abt,
            inputs=[
                "spine",
                "feature_table",
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
                "params:start_cohort",
                "params:split_cohort",
                "params:end_cohort",
                "params:audience_building.id_col",
                "params:audience_building.cohort_col",
                "params:target",
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
            func=feature_selection,
            inputs=[
                "X_train",
                "y_train",
                "params:target",
                "params:nfolds_cv"
            ],
            outputs="features_set",
            name="feature_selection_node",
            tags=["modeling-without-abt-update", "modeling-only-train"]
        ),
        node(
            func=train_oot_split,
            inputs=[
                "abt",
                "params:start_cohort",
                "params:split_cohort",
                "params:end_cohort",
                "params:audience_building.id_col",
                "params:audience_building.cohort_col",
                "params:target",
                "features_set"
            ],
            outputs=[
                "X_final_train",
                "y_final_train",
                "id_model_final_train",
                "X_final_test_oot",
                "y_final_test_oot",
                "id_model_final_test_oot",
            ],
            name="final_train_oot_split_node",
            tags=["modeling-without-abt-update"]
        ),
        node(
            func=hyperparameters_tuning,
            inputs=[
                "X_final_train",
                "y_final_train",
                "params:ntrials_optimization",
                "params:target",
                "params:random_state",
                "params:nfolds_cv"
            ],
            outputs="best_hyperparameters",
            name="hyperparameters_tuning_node",
            tags=["modeling-without-abt-update", "modeling-only-train"]
        ),
        node(
            func=ratings_optimization,
            inputs=[
                "X_final_train",
                "y_final_train",
                "best_hyperparameters",
                "params:nratings",
                "params:target",
                "params:random_state",
                "params:nfolds_cv"
            ],
            outputs="ratings_limits",
            name="ratings_optimization_node",
            tags=["modeling-without-abt-update", "modeling-only-train"]
        ),
        node(
            func=train_final_model,
            inputs=[
                "X_final_train",
                "y_final_train",
                "best_hyperparameters",
                "ratings_limits",
                "params:random_state",
                "params:nfolds_cv"
            ],
            outputs="custom_lgbm_model",
            name="train_lgbm_node",
            tags=["modeling-without-abt-update", "modeling-only-train"]
        ),
        node(
            func=model_results,
            inputs=[
                "custom_lgbm_model",
                "X_final_train",
                "y_final_train",
                "id_model_final_train",
                "X_final_test_oot",
                "y_final_test_oot",
                "id_model_final_test_oot",
            ],
            outputs=["results_train", "results_test_oot"],
            name="model_results_node",
            tags=["modeling-without-abt-update", "modeling-only-train"]
        ),
    ], tags=["modeling-without-preprocess"])

    pipe_modeling = pipeline(
        pipe_template,
        namespace='modeling',
        parameters={
            "params:audience_building.id_col",
            "params:audience_building.cohort_col",
            "params:random_state"
        }
    )

    return pipe_modeling
