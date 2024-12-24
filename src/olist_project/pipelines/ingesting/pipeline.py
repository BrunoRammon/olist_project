"""
This is a boilerplate pipeline 'ingesting'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import ingest

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=ingest,
            inputs=["params:dataset_schemas"],
            outputs=[
                'raw_customers', 'raw_orders',
                'raw_order_items','raw_products',
                'raw_geolocation', 'raw_order_payments',
                'raw_order_reviews','raw_sellers',
            ],
            name="ingest_datasets_node"
        )
    ])
