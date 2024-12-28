"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.10
"""

import pandas as pd
from geopy import distance
import numpy as np

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
            df_orders[["order_id", "customer_id",
                       "order_purchase_timestamp",
                       "order_status"]]
            .drop_duplicates(),
            on="order_id",
            how='left',
            validate='m:1'
        )
        .assign(
            cohort_info = lambda df: (
                df.order_purchase_timestamp.dt.strftime("%Y%m").astype('Int64')
            ),
            days_to_post = lambda df: (
                (df.shipping_limit_date - df.order_purchase_timestamp).dt.days
            )
        )
        .query('seller_id.notna()')
        .drop(columns=['_merge','order_item_id',
                       'shipping_limit_date'])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_pre

def preprocessing_payments(df_order_payments: pd.DataFrame,
                           df_pre_order_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_order_payments
        .merge(
            df_pre_order_items[['seller_id','order_id',
                                "cohort_info", "order_status"]]
            .drop_duplicates(),
            on=['order_id'],
            how='left',
            validate='m:m',
            indicator=True
        )
        .query('seller_id.notna()')
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = (
        df_pre
        .drop(columns=['_merge'])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_pre

def preprocessing_reviews(df_order_reviews: pd.DataFrame, 
                          df_pre_order_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_order_reviews
        .merge(
            df_pre_order_items[['seller_id','order_id','order_status',
                                "order_purchase_timestamp"]]
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
            ),
            days_to_review = lambda df: (
                (df.review_answer_timestamp-df.order_purchase_timestamp).dt.days
            ),
            days_to_sent_survey = lambda df: (
                (df.review_creation_date-df.order_purchase_timestamp).dt.days
            )
        )
        .query('seller_id.notna()')
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = (
        df_pre
        .drop(columns=['_merge','order_purchase_timestamp',
                       'review_creation_date',
                       'review_answer_timestamp'])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_pre

def preprocessing_customers(df_customers: pd.DataFrame,
                            df_pre_orders_items: pd.DataFrame)-> pd.DataFrame:
    """
    """

    df_pre = (
        df_customers
        .merge(
            df_pre_orders_items[["order_id", "customer_id", "cohort_info",
                                 "order_status", "seller_id"]]
            .drop_duplicates(),
            on="customer_id",
            how='left',
            validate='1:m',
            indicator=True
        )
        .assign(
            customer_zip_code_prefix = lambda df: df.customer_zip_code_prefix.str.zfill(5),
        )
        .query('seller_id.notna()')
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = (
        df_pre
        .drop(columns=['_merge','customer_id'])
        .drop_duplicates()
        .reset_index(drop=True)
    )

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
    )
    assert all(df_pre['_merge'].astype("string").fillna("nulo") == "both")
    df_pre = (
        df_pre
        .drop(columns=['_merge',"customer_id"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

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
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_pre

def preprocessing_geolocation(df_geolocation: pd.DataFrame,
                              df_pre_customers: pd.DataFrame,
                              df_pre_sellers: pd.DataFrame,
                              )-> pd.DataFrame:
    """
    """

    df_pre_geolocation = (
        df_geolocation
        .assign(
            geolocation_zip_code_prefix = lambda df: (
                df.geolocation_zip_code_prefix.str.zfill(5)
            )
        )
        .groupby('geolocation_zip_code_prefix', as_index=False)
        .agg(**{
            'mean_geolocation_lat': ('geolocation_lat','mean'),
            'mean_geolocation_lng': ('geolocation_lng','mean'),
        })
    )

    df_pre = (
        df_pre_customers
        .merge(
            df_pre_sellers[['seller_id','seller_zip_code_prefix']],
            on='seller_id',
            how='left'
        )
        [['seller_id','cohort_info', 'order_id', 'order_status',
          'seller_zip_code_prefix','customer_zip_code_prefix']]
        .merge(
            df_pre_geolocation,
            left_on=['customer_zip_code_prefix'],
            right_on=['geolocation_zip_code_prefix'],
            how='left'
        )
        .drop(columns='geolocation_zip_code_prefix')
        .merge(
            df_pre_geolocation,
            left_on=['seller_zip_code_prefix'],
            right_on=['geolocation_zip_code_prefix'],
            how='left',
            suffixes=('_customer','_seller')
        )
        .drop(columns='geolocation_zip_code_prefix')
        .assign(
            lat_long_customer = lambda df: list(zip(df.mean_geolocation_lat_customer.fillna(0),
                                                    df.mean_geolocation_lng_customer.fillna(0))),
            lat_long_seller = lambda df: list(zip(df.mean_geolocation_lat_seller.fillna(0),
                                                df.mean_geolocation_lng_seller.fillna(0))),
            distance_customer_seller = lambda df: df.apply(
                lambda x: distance.distance(x['lat_long_customer'],
                                            x['lat_long_seller']).km,
                axis=1
            )
        )
        .assign(
            distance_customer_seller = lambda df: (
                df.distance_customer_seller.where(
                    (df.lat_long_customer!=(0.0,0.0))&(df.lat_long_seller!=(0.0,0.0)),
                    np.nan
                )
            )
        )
        [['seller_id','cohort_info',
          'order_id', 'order_status',
          'distance_customer_seller',]]
    )

    return df_pre
