"""
This is a boilerplate pipeline 'monitoring'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    create_partitions,
    concat_partitions, 
    model_evaluation,
    performance_monitoring,
    generate_shap_values,
    shap_monitoring,
    prediction_monitoring
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_partitions,
            inputs="output",
            outputs="scored_history_partitioned",
            name="create_partition_scored_history_node",
        ),
        node(
            func=create_partitions,
            inputs="scoring.spine",
            outputs="target_history_partitioned",
            name="create_partition_target_history_node",
        ),
        node(
            func=concat_partitions,
            inputs="scored_history_partitioned",
            outputs="scored_history",
            name="update_scored_history_node"
        ),
        node(
            func=concat_partitions,
            inputs="target_history_partitioned",
            outputs="target_history",
            name="update_target_history_node"
        ),
        node(
            func=model_evaluation,
            inputs=["scored_history",
                    "target_history",
                    "params:modeling.target"],
            outputs="model_evaluation_by_cohort",
            name="model_evaluation_node"
        ),
        node(
            func=performance_monitoring,
            inputs=["scored_history",
                    "target_history",
                    "params:modeling.features",
                    "params:audience_building.id_col",
                    "params:audience_building.cohort_col",
                    "params:modeling.target",
                    "params:modeling.start_cohort",
                    "params:modeling.split_cohort",
                    "params:modeling.end_cohort",],
            outputs=["metrics_by_month",
                     "rating_metrics_by_month",
                     "metrics_by_period",
                     "rating_metrics_by_period",
                     "metrics_plot",
                     "rating_metrics_plot",
                     "feature_metrics_by_month",
                     "feature_metrics_by_period",
                     "feature_metrics_plot",
                     "features_iv_plot"],
            name="performance_monitoring_node"
        ),
        node(
            func=generate_shap_values,
            inputs=["scored_history",
                    "custom_lgbm_model",
                    "params:audience_building.id_col",
                    "params:audience_building.cohort_col",
                    "params:modeling.start_cohort",
                    "params:modeling.split_cohort",
                    "params:modeling.end_cohort",
                    "params:monitoring.shap.sample_size",
                    "params:random_state"],
            outputs="shap_values",
            name="generate_shap_values_node"
        ),
        node(
            func=shap_monitoring,
            inputs=["shap_values",
                    "scored_history",
                    "params:modeling.features",],
            outputs=[
                "shap_summary_plot",
                "shap_feature_importance_plot"
            ],
            name="shap_monitoring_node"
        ),
        node(
            func=prediction_monitoring,
            inputs=["scored_history",
                    "shap_values",
                    "params:modeling.features",
                    "params:audience_building.id_col",
                    "params:audience_building.cohort_col",
                    "params:modeling.start_cohort",
                    "params:modeling.split_cohort",
                    "params:modeling.end_cohort"],
            outputs=["customers_volume_by_month",
                     "customers_volume_by_month_plot",
                     "ratings_volume_by_month",
                     "ratings_volume_by_month_plot",
                     "migration_matrix_plot",
                     "features_groups_volume_by_month",
                     "features_statistics_by_month",
                     "features_statistics_plot",
                     "features_psi_plot"],
            name="prediction_monitoring_node"
        )
    ], tags=["scoring", "scoring-without-preprocess"])