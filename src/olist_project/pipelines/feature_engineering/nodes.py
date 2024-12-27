"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.10
"""

from typing import List
import pandas as pd
from tqdm import tqdm
import logging
from olist_project.utils.utils import (
    _check_dataset_granularity, _cohort_offset,
    _cohort_to_datetime
)

def _logging_info(message):
    logger = logging.getLogger(__name__)
    logger.info(message)


def _build_target_churn_for_cohort(
        df_orders: pd.DataFrame,
        id_orders_col: str,
        cohort_orders_col: str,
        cohort: int,
        performance_period: int
)-> pd.DataFrame:
    """
    """
    final_cohort = _cohort_offset(cohort,performance_period)
    df_target_churn_cohort = (
        df_orders
        .query(f"({cohort_orders_col}>={cohort}) & ({cohort_orders_col}<{final_cohort})")
        .drop_duplicates(subset=[id_orders_col])
        .assign(
            target_churn = 0,
            cohort = cohort
        )
        [[id_orders_col,'cohort','target_churn']]
    )
    return df_target_churn_cohort

def build_target(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_orders: pd.DataFrame,
        id_orders_col: str,
        cohort_orders_col: str,
        performance_period: int
):
    """
    """

    cohorts = sorted(df_audience[cohort_audience_col].unique())
    dfs_target_churn_cohort = []
    for cohort in cohorts:
        df_target_churn_cohort = _build_target_churn_for_cohort(
            df_orders, id_orders_col, cohort_orders_col,
            cohort, performance_period
        )
        dfs_target_churn_cohort.append(df_target_churn_cohort)

    df_spine = (
        pd.concat(dfs_target_churn_cohort, axis=0, ignore_index=True)
        .rename(columns={'cohort': cohort_audience_col,
                         id_orders_col: id_audience_col})
        .merge(
            df_audience[[id_audience_col,cohort_audience_col]],
            on=[id_audience_col,cohort_audience_col],
            how='right'
        )
        .assign(
            target_churn = lambda df: df.target_churn.fillna(1)
        )
        .rename(columns={'cohort': cohort_audience_col})
        .reset_index(drop=True)
        [[id_audience_col, cohort_audience_col, 'target_churn']]
    )

    _check_dataset_granularity(df_spine, [id_audience_col,cohort_audience_col],
                               raise_error=True)

    return df_spine

def _build_features_orders_for_cohort(
        df_orders: pd.DataFrame,
        id_orders_col: str,
        cohort_orders_col: str,
        cohort: int,
        time_window: int
)-> pd.DataFrame:

    initial_cohort = _cohort_offset(cohort,-time_window)
    dt_cohort = _cohort_to_datetime(cohort)
    pre_assign = {
        'order_approved_at_corrected': lambda df: (
            df.order_approved_at.where(df.order_approved_at < dt_cohort, dt_cohort)
        ),
        'order_delivered_carrier_date_corrected': lambda df: (
            df.order_delivered_carrier_date.where(df.order_delivered_carrier_date < dt_cohort,
                                                  dt_cohort)
        ),
        'order_delivered_customer_date_corrected': lambda df: (
            df.order_delivered_customer_date.where(df.order_delivered_customer_date < dt_cohort,
                                                  dt_cohort)
        ),
        'estimated_days_to_order_delivery': lambda df: (
            df.order_estimated_delivery_date-df.order_purchase_timestamp
        ).dt.days,
        'days_to_order_approval': lambda df: (
            df.order_approved_at_corrected-df.order_purchase_timestamp
        ).dt.days,
        'days_to_order_posting': lambda df: (
            df.order_delivered_carrier_date_corrected-df.order_purchase_timestamp
        ).dt.days,
        'days_to_order_delivery': lambda df: (
            df.order_delivered_customer_date_corrected-df.order_purchase_timestamp
        ).dt.days,
        'diff_days_actual_estimated_delivery': lambda df: (
            df.order_estimated_delivery_date-df.order_delivered_customer_date_corrected
        ).dt.days,
    }
    agg_function = {
        f'total_orders_m{time_window}': ('order_id','nunique'),

        f'mean_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','mean'),
        f'max_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','max'),
        f'min_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','min'),
        f'median_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','median'),
        f'std_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','std'),

        f'mean_days_to_order_approval_m{time_window}': ('days_to_order_approval','mean'),
        f'max_days_to_order_approval_m{time_window}': ('days_to_order_approval','max'),
        f'min_days_to_order_approval_m{time_window}': ('days_to_order_approval','min'),
        f'median_days_to_order_approval_m{time_window}': ('days_to_order_approval','median'),
        f'std_days_to_order_approval_m{time_window}': ('days_to_order_approval','std'),

        f'mean_days_to_order_posting_m{time_window}': ('days_to_order_posting','mean'),
        f'max_days_to_order_posting_m{time_window}': ('days_to_order_posting','max'),
        f'min_days_to_order_posting_m{time_window}': ('days_to_order_posting','min'),
        f'median_days_to_order_posting_m{time_window}': ('days_to_order_posting','median'),
        f'std_days_to_order_posting_m{time_window}': ('days_to_order_posting','std'),

        f'mean_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','mean'),
        f'max_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','max'),
        f'min_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','min'),
        f'median_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','median'),
        f'std_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','std'),

        f'mean_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','mean'),
        f'max_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','max'),
        f'min_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','min'),
        f'median_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','median'),
        f'std_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','std'),

        f'max_order_purchase_timestamp_m{time_window}': ('order_purchase_timestamp','max'),
    }
    post_assign = {
        'cohort': cohort,
        f'recencia_m{time_window}': lambda df: (
            dt_cohort - df[f'max_order_purchase_timestamp_m{time_window}']
        ).dt.days,
    }
    post_drop_cols = [
        f'max_order_purchase_timestamp_m{time_window}'
    ]
    df_orders_cohort_time_window = (
        df_orders
        .query(f'({cohort_orders_col}>={initial_cohort})&({cohort_orders_col}<{cohort})')
        .assign(**pre_assign)
        .groupby(id_orders_col, as_index=False)
        .agg(**agg_function)
        .assign(**post_assign)
        .drop(columns=post_drop_cols)
    )

    return df_orders_cohort_time_window

def build_features_orders(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_orders: pd.DataFrame,
        id_orders_col: str,
        cohort_orders_col: str,
        time_windows: List[int]
)-> pd.DataFrame:
    """
    """

    df_features = df_audience[[id_audience_col,cohort_audience_col]]
    cohorts = sorted(df_audience[cohort_audience_col].unique())
    for time_window in time_windows:
        message = f'Calculating {time_window}M features...'
        _logging_info(message)

        df_features_cohort_time_window_list = []
        for cohort in tqdm(cohorts):
            df_features_cohort_time_window = _build_features_orders_for_cohort(
                df_orders, id_orders_col, cohort_orders_col,
                cohort, time_window
            )
            df_features_cohort_time_window_list.append(df_features_cohort_time_window)

        df_features_time_window = pd.concat(df_features_cohort_time_window_list,
                                            axis=0, ignore_index=True)

        df_features = (
            df_features
            .merge(df_features_time_window,
                   on=[id_audience_col,cohort_audience_col],
                   how='left')
            .reset_index(drop=True)
        )

    return df_features
