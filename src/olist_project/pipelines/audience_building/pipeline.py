"""
This is a boilerplate pipeline 'audience_building'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    build_initial_audience,
    build_audience_filters,
    build_final_audience
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_initial_audience,
            inputs=[
                "pre_orders",
                "params:audience_building.id_col",
                "params:audience_building.cohort_info_col",
                "params:modeling.start_cohort",
                "params:modeling.end_cohort",
                "params:feature_engineering.historical_period",
            ],
            outputs='initial_audience',
            name='build_initial_audience_node'
        ),
        node(
            func=build_audience_filters,
            inputs=[
                "pre_orders",
                "params:audience_building.id_col",
                "params:audience_building.cohort_info_col",
                "params:feature_engineering.historical_period",
            ],
            outputs='flags_filters',
            name='build_audience_filters_node'
        ),
        node(
            func=build_final_audience,
            inputs=[
                "initial_audience",
                "flags_filters",
                "params:audience_building.filters_specs",
                "params:audience_building.id_col",
                "params:audience_building.cohort_col",
            ],
            outputs='modeling_audience',
            name='build_final_audience'
        ),
    ])
