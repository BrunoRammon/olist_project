"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    preprocessing_customers,
    preprocessing_items,
    preprocessing_orders,
    preprocessing_payments,
    preprocessing_reviews,
    preprocessing_sellers
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocessing_customers,
            inputs=["raw_customers",
                    "raw_orders",
                    "raw_order_items"],
            outputs='pre_customers',
            name="preprocessing_customers_node"
        ),
        node(
            func=preprocessing_items,
            inputs=["raw_order_items",
                    "raw_products",
                    "raw_orders"],
            outputs='pre_order_items',
            name="preprocessing_order_items_node"
        ),
        node(
            func=preprocessing_orders,
            inputs=["raw_orders",
                    "raw_order_items"],
            outputs='pre_orders',
            name="preprocessing_orders_node"
        ),
        node(
            func=preprocessing_payments,
            inputs=[
                "raw_order_payments",
                "raw_orders",
                "raw_order_items",
            ],
            outputs='pre_order_payments',
            name="preprocessing_order_payments_node"
        ),
        node(
            func=preprocessing_reviews,
            inputs=[
                "raw_order_reviews",
                "raw_order_items",
            ],
            outputs='pre_order_reviews',
            name="preprocessing_order_reviews_node"
        ),
        node(
            func=preprocessing_sellers,
            inputs="raw_sellers",
            outputs='pre_sellers',
            name="preprocessing_sellers_node"
        ),
    ])