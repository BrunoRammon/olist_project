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
            inputs=None,
            outputs=[
                'olist_customers_dataset',
                'olist_geolocation_dataset',
                'olist_order_items_dataset',
                'olist_order_payments_dataset',
                'olist_order_reviews_dataset',
                'olist_orders_dataset',
                'olist_products_dataset',
                'olist_sellers_dataset',
                'product_category_name_translation'
            ],
            name="ingest_datasets_node"
        )
    ])
