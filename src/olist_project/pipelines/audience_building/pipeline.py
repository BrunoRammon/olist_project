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
    pipe_template = pipeline([
        node(
            func=build_initial_audience,
            inputs=[
                "pre_orders",
                "params:audience_building.id_col",
                "params:audience_building.cohort_info_col",
                "params:namespace.start_cohort",
                "params:namespace.end_cohort",
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
            outputs='audience',
            name='build_final_audience'
        ),
    ])

    namespace = "modeling"
    modeling_pipe = pipeline(
        pipe=pipe_template,
        parameters={
            "params:namespace.start_cohort": f"params:{namespace}.start_cohort",
            "params:namespace.end_cohort": f"params:{namespace}.end_cohort",
            "params:feature_engineering.historical_period": "params:feature_engineering.historical_period",
        },
        namespace=namespace,
        tags=[f"{namespace}-without-preprocess"]
    )
    namespace = "scoring"
    scoring_pipe = pipeline(
        pipe=pipe_template,
        parameters={
            "params:namespace.start_cohort": f"params:{namespace}.start_cohort",
            "params:namespace.end_cohort": f"params:{namespace}.end_cohort",
            "params:feature_engineering.historical_period": "params:feature_engineering.historical_period",
        },
        namespace=namespace,
        tags=[f"{namespace}-without-preprocess"]
    )

    return modeling_pipe + scoring_pipe
