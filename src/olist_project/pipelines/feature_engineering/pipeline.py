"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import build_target, build_features_orders


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_target,
            inputs=[
                "modeling_audience",
                "params:audience_building.id_col",
                "params:audience_building.cohort_col",
                "pre_orders",
                "params:feature_engineering.orders.id_col",
                "params:feature_engineering.orders.cohort_info_col",
                "params:feature_engineering.performance_period",
            ],
            outputs="modeling_spine",
            name="build_target_node"
        ),
        node(
            func=build_features_orders,
            inputs=[
                "modeling_audience",
                "params:audience_building.id_col",
                "params:audience_building.cohort_col",
                "pre_orders",
                "params:feature_engineering.orders.id_col",
                "params:feature_engineering.orders.cohort_info_col",
                "params:feature_engineering.time_windows",
            ],
            outputs="feat_modeling_orders",
            name="build_features_orders_node"
        )
    ])
