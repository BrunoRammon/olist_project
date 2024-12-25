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
            df_orders[["order_id", "order_approved_at"]]
            .drop_duplicates(),
            on="order_id",
            how='left',
            validate='m:1'
        )
        .merge(
            df_order_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='m:m',
            indicator=True
        )
        .assign(
            cohort_info = lambda df: (
                df.order_approved_at.dt.strftime("%Y%m").astype('Int64')
            )
        )
        .query('seller_id.notna()')
        .reset_index(drop=True)
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = df_pre.drop(columns=['_merge',"order_id","order_approved_at"])

    return df_pre

def preprocessing_reviews(df_order_reviews: pd.DataFrame,
                          df_order_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_order_reviews
        .merge(
            df_order_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='m:m',
            indicator=True
        )
        .assign(
            cohort_info = lambda df: (
                df.review_answer_timestamp.dt.strftime("%Y%m").astype('Int64')
            ),
            delay_answer_review = lambda df: (
                (df.review_answer_timestamp - df.review_creation_date).dt.days
            )
        )
        .query('seller_id.notna()')
        .reset_index(drop=True)
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
            df_orders[["order_id", "order_purchase_timestamp"]]
            .drop_duplicates(),
            on="order_id",
            how='left',
            validate='m:1'
        )
        .assign(
            cohort_info = lambda df: (
                df.order_purchase_timestamp.dt.strftime("%Y%m").astype('Int64')
            )
        )
        .query('seller_id.notna()')
        .reset_index(drop=True)
    )
    df_pre = df_pre.drop(columns=['order_id', '_merge','order_item_id',
                                  'order_purchase_timestamp'])

    return df_pre

def preprocessing_customers(df_customers: pd.DataFrame,
                            df_orders: pd.DataFrame,
                            df_orders_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_customers
        .merge(
            df_orders[["order_id", "customer_id",
                       "order_purchase_timestamp"]]
            .drop_duplicates(),
            on="customer_id",
            how='left',
            validate='1:1'
        )
        .merge(
            df_orders_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='1:m',
            indicator=True
        )
        .assign(
            customer_zip_code_prefix = lambda df: df.customer_zip_code_prefix.str.zfill(5),
            cohort_info = lambda df: (
                df.order_purchase_timestamp.dt.strftime("%Y%m").astype('Int64')
            )
        )
        .query('seller_id.notna()')
        .reset_index(drop=True)
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = df_pre.drop(columns=['_merge','customer_id',
                                  'order_id', 'order_purchase_timestamp'])

    return df_pre

def preprocessing_orders(df_orders: pd.DataFrame,
                         df_orders_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_orders
        .merge(
            df_orders_items[['seller_id','order_id']]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='1:m',
            indicator=True
        )
        .assign(
            cohort_info = lambda df: (
                df.order_purchase_timestamp.dt.strftime("%Y%m").astype('Int64')
            )
        )
        .query('seller_id.notna()')
        .reset_index(drop=True)
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = df_pre.drop(columns=['_merge',"customer_id"])

    return df_pre

def preprocessing_sellers(df_sellers: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_sellers
        .assign(
            seller_zip_code_prefix = lambda df: df.seller_zip_code_prefix.str.zfill(5)
        )
        .query('seller_id.notna()')
        .reset_index(drop=True)
    )

    return df_pre
