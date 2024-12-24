"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.10
"""

import pandas as pd

def preprocessing_payments(df_order_payments: pd.DataFrame,
                           df_orders: pd.DataFrame,
                           df_order_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_order_payments
        .merge(
            df_orders[["order_id", "order_status", "order_purchase_timestamp"]]
            .drop_duplicates(),
            on="order_id",
            how='left',
            validate='m:1'
        )
        .query("order_status=='delivered'")
        .drop(columns="order_status")
        .merge(
            df_order_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='m:m',
            indicator=True
        )
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = df_pre.drop(columns=['_merge'])

    return df_pre

def preprocessing_reviews(df_order_reviews: pd.DataFrame,
                          df_orders: pd.DataFrame,
                          df_order_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_order_reviews
        .merge(
            df_orders[["order_id", "order_status"]]
            .drop_duplicates(),
            on="order_id",
            how='left',
            validate='m:1'
        )
        .query("order_status=='delivered'")
        .drop(columns="order_status")
        .merge(
            df_order_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='m:m',
            indicator=True
        )
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = df_pre.drop(columns=['_merge','order_id'])

    return df_pre

def preprocessing_items(df_order_items: pd.DataFrame,
                        df_products: pd.DataFrame,
                        df_orders: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_order_items
        .merge(df_products,
            on=['product_id'],
            validate='m:1',
            how='left',
            indicator=True)
    )

    assert all(df_pre['_merge']=="both")
    # o dataset resultante deve ter o mesmo tamnho
    assert len(df_pre) == len(df_order_items)
    df_pre = (
        df_pre
        .merge(
            df_orders[["order_id", "order_status", "order_purchase_timestamp"]]
            .drop_duplicates(),
            on="order_id",
            how='left',
            validate='m:1'
        )
        .query("order_status=='delivered'")
    )
    df_pre = df_pre.drop(columns=['order_status','order_id', '_merge','order_item_id'])

    return df_pre

def preprocessing_customers(df_customers: pd.DataFrame,
                            df_orders: pd.DataFrame,
                            df_orders_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_customers
        .merge(
            df_orders[["order_id", "customer_id", "order_status",
                       "order_purchase_timestamp"]]
            .drop_duplicates(),
            on="customer_id",
            how='left',
            validate='1:1'
        )
        .query("order_status=='delivered'")
        .merge(
            df_orders_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='1:m',
            indicator=True
        )
        .assign(
            customer_zip_code_prefix = lambda df: df.customer_zip_code_prefix.str.zfill(5)
        )
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = df_pre.drop(columns=['_merge', 'order_status','customer_id', 'order_id'])

    return df_pre

def preprocessing_orders(df_orders: pd.DataFrame,
                         df_orders_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_orders
        .query("order_status=='delivered'")
        .merge(
            df_orders_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='1:m',
            indicator=True
        )
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = df_pre.drop(columns=['_merge',"customer_id"])

    return df_pre

def preprocessing_sellers(df_sellers: pd.DataFrame,
                          df_orders: pd.DataFrame,
                          df_orders_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_sellers
        .merge(
            df_orders_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['seller_id'],
            how='left',
            validate='m:m',
        )
        .merge(
            df_orders[["order_id", "order_status"]]
            .drop_duplicates(),
            on="order_id",
            how='left',
            validate='m:1'
        )
        .query("order_status=='delivered'")
        .drop_duplicates(['seller_id'])
        .drop(columns=["order_status","order_id"])
        .assign(
            seller_zip_code_prefix = lambda df: df.seller_zip_code_prefix.str.zfill(5)
        )
    )

    return df_pre
