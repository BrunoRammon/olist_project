"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.10
"""

from typing import List, Dict, Callable, Tuple
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
from olist_project.utils.utils import (
    _check_dataset_granularity, _cohort_offset,
    _cohort_to_datetime, ConsistenceDataError
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

def _compute_slope(gr, value_col):
    len_col = len(gr[value_col])
    x = np.linspace(1,len_col,num=len_col)
    y = gr[value_col]
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator if denominator != 0 else np.nan

def _build_slope_feature_for_cohort_time_window(df_historical: pd.DataFrame,
                                                id_col: str,
                                                cohort_col: str,
                                                current_cohort: int,
                                                time_window: int,
                                                value_col: str,
                                                agg_value_func: str,
                                                fillna_value: float=0.0,
                                                )-> pd.DataFrame:
    initial_cohort = _cohort_offset(current_cohort,-time_window)
    return (
        df_historical
        .query(f'({cohort_col}>={initial_cohort})&({cohort_col}<{current_cohort})')
        .groupby([id_col,cohort_col],as_index=False)
        .agg(
            total = (value_col, agg_value_func)
        )
        .sort_values(cohort_col)
        .pivot(index=id_col,columns=cohort_col,values='total')
        .fillna(fillna_value)
        .reset_index()
        .melt(id_vars=id_col, value_name='total')
        .groupby(id_col)
        .apply(_compute_slope,value_col='total', include_groups=False)
        .reset_index(name=f'slope_{agg_value_func}_{value_col}_m{time_window}')
        .assign(
            cohort=current_cohort
        )
    )

def _build_features_for_cohort_time_window(df_historical: pd.DataFrame,
                                           id_col: str,
                                           cohort_col: str,
                                           current_cohort: int,
                                           time_window: int,
                                           pre_assign: Dict[str,Callable],
                                           agg_functions: Dict[str,Tuple[str,str]],
                                           post_assign: Dict[str,Callable],
                                           post_drop_cols: List[str])-> pd.DataFrame:
    post_assign_cp = dict(post_assign)
    post_assign_cp['cohort'] = current_cohort
    initial_cohort = _cohort_offset(current_cohort,-time_window)
    return (
        df_historical
        .query(f'({cohort_col}>={initial_cohort})&({cohort_col}<{current_cohort})')
        .assign(**pre_assign)
        .groupby(id_col, as_index=False)
        .agg(**agg_functions)
        .assign(**post_assign_cp)
        .drop(columns=post_drop_cols)
    )

def _build_features_orders_for_cohort(
        df_orders: pd.DataFrame,
        id_orders_col: str,
        cohort_orders_col: str,
        current_cohort: int,
        time_window: int
)-> pd.DataFrame:

    dt_cohort = _cohort_to_datetime(current_cohort)
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
    agg_functions = {
        f'total_orders_m{time_window}': ('order_id','nunique'),

        f'nunique_cohorts_m{time_window}': (f'{cohort_orders_col}','nunique'),

        f'mean_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','mean'),
        f'std_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','std'),
        f'max_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','max'),
        f'min_estimated_days_to_order_delivery_m{time_window}': ('estimated_days_to_order_delivery','min'),

        f'mean_days_to_order_approval_m{time_window}': ('days_to_order_approval','mean'),
        f'std_days_to_order_approval_m{time_window}': ('days_to_order_approval','std'),
        f'max_days_to_order_approval_m{time_window}': ('days_to_order_approval','max'),
        f'min_days_to_order_approval_m{time_window}': ('days_to_order_approval','min'),

        f'mean_days_to_order_posting_m{time_window}': ('days_to_order_posting','mean'),
        f'std_days_to_order_posting_m{time_window}': ('days_to_order_posting','std'),
        f'max_days_to_order_posting_m{time_window}': ('days_to_order_posting','max'),
        f'min_days_to_order_posting_m{time_window}': ('days_to_order_posting','min'),

        f'mean_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','mean'),
        f'std_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','std'),
        f'max_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','max'),
        f'min_days_to_order_delivery_m{time_window}': ('days_to_order_delivery','min'),

        f'mean_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','mean'),
        f'std_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','std'),
        f'max_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','max'),
        f'min_diff_days_actual_estimated_delivery_m{time_window}': ('diff_days_actual_estimated_delivery','min'),

        f'max_order_purchase_timestamp_m{time_window}': ('order_purchase_timestamp','max'),
    }

    post_assign = {
        f'recency_m{time_window}': lambda df: (
            dt_cohort - df[f'max_order_purchase_timestamp_m{time_window}']
        ).dt.days,
        f'mean_frequency_m{time_window}': lambda df: (
            df[f'total_orders_m{time_window}']/time_window
        ),
        f'rate_nunique_cohorts_m{time_window}': lambda df: (
            df[f'nunique_cohorts_m{time_window}']/time_window
        ),
    }

    post_drop_cols = [
        f'max_order_purchase_timestamp_m{time_window}'
    ]

    df_orders_cohort_time_window = (
        _build_features_for_cohort_time_window(df_orders,
                                               id_orders_col,
                                               cohort_orders_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
        .merge(
            _build_slope_feature_for_cohort_time_window(
                df_orders,
                id_orders_col,
                cohort_orders_col,
                current_cohort,
                time_window,
                'order_id',
                'nunique',
                fillna_value=0.0
            ),
            on=[id_orders_col,'cohort'],
            how='left'
        )                 
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
            .merge(df_features_time_window
                   .rename(columns={'cohort': cohort_audience_col,
                                    id_orders_col: id_audience_col}),
                   on=[id_audience_col,cohort_audience_col],
                   how='left')
            .reset_index(drop=True)
        )

    _check_dataset_granularity(df_features, [id_audience_col,cohort_audience_col],
                               raise_error=True)

    return df_features

def _build_features_items_for_cohort(
        df_items: pd.DataFrame,
        id_items_col: str,
        cohort_items_col: str,
        current_cohort: int,
        time_window: int
)-> pd.DataFrame:

    pre_assign = {}
    agg_functions = {
        f'nunique_products_m{time_window}': ('product_id','nunique'),
        f'count_total_unq_products_m{time_window}': ('product_id','count'),

        f'sum_price_products_m{time_window}': ('price','sum'),
        f'mean_price_products_m{time_window}': ('price','mean'),
        f'std_price_products_m{time_window}': ('price','std'),
        f'max_price_products_m{time_window}': ('price','max'),
        f'min_price_products_m{time_window}': ('price','min'),

        f'sum_freight_value_products_m{time_window}': ('freight_value','sum'),
        f'mean_freight_value_products_m{time_window}': ('freight_value','mean'),
        f'std_freight_value_products_m{time_window}': ('freight_value','std'),
        f'max_freight_value_products_m{time_window}': ('freight_value','max'),
        f'min_freight_value_products_m{time_window}': ('freight_value','min'),

        f'mean_product_name_lenght_products_m{time_window}': ('product_name_lenght','mean'),
        f'std_product_name_lenght_products_m{time_window}': ('product_name_lenght','std'),
        f'max_product_name_lenght_products_m{time_window}': ('product_name_lenght','max'),
        f'min_product_name_lenght_products_m{time_window}': ('product_name_lenght','min'),

        f'mean_product_description_lenght_products_m{time_window}': ('product_description_lenght','mean'),
        f'std_product_description_lenght_products_m{time_window}': ('product_description_lenght','std'),
        f'max_product_description_lenght_products_m{time_window}': ('product_description_lenght','max'),
        f'min_product_description_lenght_products_m{time_window}': ('product_description_lenght','min'),

        f'mean_product_photos_qty_products_m{time_window}': ('product_photos_qty','mean'),
        f'std_product_photos_qty_products_m{time_window}': ('product_photos_qty','std'),
        f'max_product_photos_qty_products_m{time_window}': ('product_photos_qty','max'),
        f'min_product_photos_qty_products_m{time_window}': ('product_photos_qty','min'),

        f'mean_product_weight_g_products_m{time_window}': ('product_weight_g','mean'),
        f'std_product_weight_g_products_m{time_window}': ('product_weight_g','std'),
        f'max_product_weight_g_products_m{time_window}': ('product_weight_g','max'),
        f'min_product_weight_g_products_m{time_window}': ('product_weight_g','min'),

        f'mean_product_length_cm_products_m{time_window}': ('product_length_cm','mean'),
        f'std_product_length_cm_products_m{time_window}': ('product_length_cm','std'),
        f'max_product_length_cm_products_m{time_window}': ('product_length_cm','max'),
        f'min_product_length_cm_products_m{time_window}': ('product_length_cm','min'),

        f'mean_product_height_cm_products_m{time_window}': ('product_height_cm','mean'),
        f'std_product_height_cm_products_m{time_window}': ('product_height_cm','std'),
        f'max_product_height_cm_products_m{time_window}': ('product_height_cm','max'),
        f'min_product_height_cm_products_m{time_window}': ('product_height_cm','min'),

        f'mean_product_width_cm_products_m{time_window}': ('product_width_cm','mean'),
        f'std_product_width_cm_products_m{time_window}': ('product_width_cm','std'),
        f'max_product_width_cm_products_m{time_window}': ('product_width_cm','max'),
        f'min_product_width_cm_products_m{time_window}': ('product_width_cm','min'),

        f'mean_days_to_post_products_m{time_window}': ('days_to_post','mean'),
        f'std_days_to_post_products_m{time_window}': ('days_to_post','std'),
        f'max_days_to_post_products_m{time_window}': ('days_to_post','max'),
        f'min_days_to_post_products_m{time_window}': ('days_to_post','min'),
    }

    post_assign = {}

    post_drop_cols = []

    df_items_cohort_time_window = (
        _build_features_for_cohort_time_window(df_items,
                                               id_items_col,
                                               cohort_items_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
        .merge(
            _build_slope_feature_for_cohort_time_window(
                df_items,
                id_items_col,
                cohort_items_col,
                current_cohort,
                time_window,
                'price',
                'sum',
                fillna_value=0.0
            ),
            on=[id_items_col,'cohort'],
            how='left'
        )
        .merge(
            _build_slope_feature_for_cohort_time_window(
                df_items,
                id_items_col,
                cohort_items_col,
                current_cohort,
                time_window,
                'product_id',
                'count',
                fillna_value=0.0
            ),
            on=[id_items_col,'cohort'],
            how='left'
        )
    )

    return df_items_cohort_time_window

def build_features_items(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_order_items: pd.DataFrame,
        id_order_items_col: str,
        cohort_order_items_col: str,
        time_windows: List[int]
)-> pd.DataFrame:
    """
    """

    df_order_items = df_order_items.query('order_status=="delivered"')
    df_features = df_audience[[id_audience_col,cohort_audience_col]]
    cohorts = sorted(df_audience[cohort_audience_col].unique())
    for time_window in time_windows:
        message = f'Calculating {time_window}M features...'
        _logging_info(message)

        df_features_cohort_time_window_list = []
        for cohort in tqdm(cohorts):
            df_features_cohort_time_window = _build_features_items_for_cohort(
                df_order_items, id_order_items_col, cohort_order_items_col,
                cohort, time_window
            )
            df_features_cohort_time_window_list.append(df_features_cohort_time_window)

        df_features_time_window = pd.concat(df_features_cohort_time_window_list,
                                            axis=0, ignore_index=True)

        df_features = (
            df_features
            .merge(df_features_time_window
                   .rename(columns={'cohort': cohort_audience_col,
                                    id_order_items_col: id_audience_col}),
                   on=[id_audience_col,cohort_audience_col],
                   how='left')
            .reset_index(drop=True)
        )

    _check_dataset_granularity(df_features, [id_audience_col,cohort_audience_col],
                               raise_error=True)

    return df_features


def _build_features_reviews_for_cohort(
        df_reviews: pd.DataFrame,
        id_reviews_col: str,
        cohort_reviews_col: str,
        current_cohort: int,
        time_window: int
)-> pd.DataFrame:

    pre_assign = {}
    agg_functions = {
        f'nunique_review_m{time_window}': ('review_id','nunique'),

        f'mean_review_score_m{time_window}': ('review_score','mean'),
        f'std_review_score_m{time_window}': ('review_score','std'),
        f'max_review_score_m{time_window}': ('review_score','max'),
        f'min_review_score_m{time_window}': ('review_score','min'),

        f'mean_days_to_review_m{time_window}': ('days_to_review','mean'),
        f'std_days_to_review_m{time_window}': ('days_to_review','std'),
        f'max_days_to_review_m{time_window}': ('days_to_review','max'),
        f'min_days_to_review_m{time_window}': ('days_to_review','min'),

        f'mean_delay_answer_review_m{time_window}': ('delay_answer_review','mean'),
        f'std_delay_answer_review_m{time_window}': ('delay_answer_review','std'),
        f'max_delay_answer_review_m{time_window}': ('delay_answer_review','max'),
        f'min_delay_answer_review_m{time_window}': ('delay_answer_review','min'),

        f'mean_days_to_sent_survey_m{time_window}': ('days_to_sent_survey','mean'),
        f'std_days_to_sent_survey_m{time_window}': ('days_to_sent_survey','std'),
        f'max_days_to_sent_survey_m{time_window}': ('days_to_sent_survey','max'),
        f'min_days_to_sent_survey_m{time_window}': ('days_to_sent_survey','min'),
    }

    post_assign = {}

    post_drop_cols = []

    df_reviews_cohort_time_window = (
        _build_features_for_cohort_time_window(df_reviews,
                                               id_reviews_col,
                                               cohort_reviews_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
    )

    return df_reviews_cohort_time_window

def build_features_reviews(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_order_reviews: pd.DataFrame,
        id_order_reviews_col: str,
        cohort_order_reviews_col: str,
        time_windows: List[int]
)-> pd.DataFrame:
    """
    """
    df_order_reviews = df_order_reviews.query('order_status=="delivered"')
    df_features = df_audience[[id_audience_col,cohort_audience_col]]
    cohorts = sorted(df_audience[cohort_audience_col].unique())
    for time_window in time_windows:
        message = f'Calculating {time_window}M features...'
        _logging_info(message)

        df_features_cohort_time_window_list = []
        for cohort in tqdm(cohorts):
            df_features_cohort_time_window = _build_features_reviews_for_cohort(
                df_order_reviews, id_order_reviews_col, cohort_order_reviews_col,
                cohort, time_window
            )
            df_features_cohort_time_window_list.append(df_features_cohort_time_window)

        df_features_time_window = pd.concat(df_features_cohort_time_window_list,
                                            axis=0, ignore_index=True)

        df_features = (
            df_features
            .merge(df_features_time_window
                   .rename(columns={'cohort': cohort_audience_col,
                                    id_order_reviews_col: id_audience_col}),
                   on=[id_audience_col,cohort_audience_col],
                   how='left')
            .reset_index(drop=True)
        )

    _check_dataset_granularity(df_features, [id_audience_col,cohort_audience_col],
                               raise_error=True)

    return df_features

def _build_features_payments_for_cohort(
        df_payments: pd.DataFrame,
        id_payments_col: str,
        cohort_payments_col: str,
        current_cohort: int,
        time_window: int
)-> pd.DataFrame:

    pre_assign = {}
    agg_functions = {

        f'mean_value_m{time_window}': ('payment_value', 'mean'),
        f'std_value_m{time_window}': ('payment_value', 'std'),
        f'max_value_m{time_window}': ('payment_value', 'max'),
        f'min_value_m{time_window}': ('payment_value', 'min'),

        f'max_payment_sequential_m{time_window}': ('payment_sequential', 'max'),
        f'median_payment_sequential_m{time_window}': ('payment_sequential', 'median'),
        f'min_payment_sequential_m{time_window}': ('payment_sequential', 'min'),
    }

    post_assign = {}

    post_drop_cols = []

    df_payments_all_cohort_time_window = (
        _build_features_for_cohort_time_window(df_payments,
                                               id_payments_col,
                                               cohort_payments_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
    )

    pre_assign = {}
    agg_functions = {
        f'count_payment_credit_card_m{time_window}': ('payment_value', 'count'),

        f'mean_installments_credit_card_m{time_window}': ('payment_installments', 'mean'),
        f'std_installments_credit_card_m{time_window}': ('payment_installments', 'std'),
        f'max_installments_credit_card_m{time_window}': ('payment_installments', 'max'),
        f'min_installments_credit_card_m{time_window}': ('payment_installments', 'min'),

        f'sum_value_credit_card_m{time_window}': ('payment_value', 'sum'),
        f'mean_value_credit_card_m{time_window}': ('payment_value', 'mean'),
        f'std_value_credit_card_m{time_window}': ('payment_value', 'std'),
        f'max_value_credit_card_m{time_window}': ('payment_value', 'max'),
        f'min_value_credit_card_m{time_window}': ('payment_value', 'min'),
    }

    post_assign = {}

    post_drop_cols = []
    df_payments_credit_card_cohort_time_window = (
        _build_features_for_cohort_time_window(df_payments
                                               .query('payment_type=="credit_card"'),
                                               id_payments_col,
                                               cohort_payments_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
    )

    pre_assign = {}
    agg_functions = {
        f'count_payment_not_credit_card_m{time_window}': ('payment_value', 'count'),

        f'sum_value_not_credit_card_m{time_window}': ('payment_value', 'sum'),
        f'mean_value_not_credit_card_m{time_window}': ('payment_value', 'mean'),
        f'std_value_not_credit_card_m{time_window}': ('payment_value', 'std'),
        f'max_value_not_credit_card_m{time_window}': ('payment_value', 'max'),
        f'min_value_not_credit_card_m{time_window}': ('payment_value', 'min'),
    }

    post_assign = {}

    post_drop_cols = []
    df_payments_not_credit_card_cohort_time_window = (
        _build_features_for_cohort_time_window(df_payments
                                               .query('payment_type!="credit_card"'),
                                               id_payments_col,
                                               cohort_payments_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
    )

    df_payments_cohort_time_window =  (
        df_payments_all_cohort_time_window
        .merge(
            df_payments_credit_card_cohort_time_window,
            on=[id_payments_col,'cohort'],
            how='left'
        )
        .merge(
            df_payments_not_credit_card_cohort_time_window,
            on=[id_payments_col,'cohort'],
            how='left'
        )
    )

    return df_payments_cohort_time_window

def build_features_payments(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_order_payments: pd.DataFrame,
        id_order_payments_col: str,
        cohort_order_payments_col: str,
        time_windows: List[int]
)-> pd.DataFrame:
    """
    """
    df_order_payments = df_order_payments.query('order_status=="delivered"')
    df_features = df_audience[[id_audience_col,cohort_audience_col]]
    cohorts = sorted(df_audience[cohort_audience_col].unique())
    for time_window in time_windows:
        message = f'Calculating {time_window}M features...'
        _logging_info(message)

        df_features_cohort_time_window_list = []
        for cohort in tqdm(cohorts):
            df_features_cohort_time_window = _build_features_payments_for_cohort(
                df_order_payments, id_order_payments_col, cohort_order_payments_col,
                cohort, time_window
            )
            df_features_cohort_time_window_list.append(df_features_cohort_time_window)

        df_features_time_window = pd.concat(df_features_cohort_time_window_list,
                                            axis=0, ignore_index=True)

        df_features = (
            df_features
            .merge(df_features_time_window
                   .rename(columns={'cohort': cohort_audience_col,
                                    id_order_payments_col: id_audience_col}),
                   on=[id_audience_col,cohort_audience_col],
                   how='left')
            .reset_index(drop=True)
        )

    _check_dataset_granularity(df_features, [id_audience_col,cohort_audience_col],
                               raise_error=True)

    return df_features

def _build_features_customers_for_cohort(
        df_customers: pd.DataFrame,
        id_customers_col: str,
        cohort_customers_col: str,
        current_cohort: int,
        time_window: int
)-> pd.DataFrame:

    pre_assign = {}
    agg_functions = {
        f'nunique_customer_unique_id_m{time_window}': ('customer_unique_id', 'nunique'),
        f'nunique_customer_zip_code_prefix_dig_1_m{time_window}': ('customer_zip_code_prefix_dig_1', 'nunique'),
        f'nunique_customer_zip_code_prefix_dig_2_m{time_window}': ('customer_zip_code_prefix_dig_2', 'nunique'),
        f'nunique_customer_zip_code_prefix_dig_3_m{time_window}': ('customer_zip_code_prefix_dig_3', 'nunique'),
        f'nunique_customer_zip_code_prefix_dig_4_m{time_window}': ('customer_zip_code_prefix_dig_4', 'nunique'),
        f'nunique_customer_zip_code_prefix_dig_5_m{time_window}': ('customer_zip_code_prefix_dig_5', 'nunique'),
        f'nunique_customer_state_m{time_window}': ('customer_state', 'nunique'),

    }

    post_assign = {}

    post_drop_cols = []

    df_customers_cohort_time_window = (
        _build_features_for_cohort_time_window(df_customers,
                                               id_customers_col,
                                               cohort_customers_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
    )

    return df_customers_cohort_time_window

def build_features_customers(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_customers: pd.DataFrame,
        id_customers_col: str,
        cohort_customers_col: str,
        time_windows: List[int]
)-> pd.DataFrame:
    """
    """
    df_customers = (
        df_customers
        .query('order_status=="delivered"')
        .assign(
            customer_zip_code_prefix_dig_1 = lambda df: df.customer_zip_code_prefix.str.slice(0,1),
            customer_zip_code_prefix_dig_2 = lambda df: df.customer_zip_code_prefix.str.slice(0,2),
            customer_zip_code_prefix_dig_3 = lambda df: df.customer_zip_code_prefix.str.slice(0,3),
            customer_zip_code_prefix_dig_4 = lambda df: df.customer_zip_code_prefix.str.slice(0,4),
            customer_zip_code_prefix_dig_5 = lambda df: df.customer_zip_code_prefix,
        )
        .drop(columns=['customer_zip_code_prefix'])
    )
    df_features = df_audience[[id_audience_col,cohort_audience_col]]
    cohorts = sorted(df_audience[cohort_audience_col].unique())
    for time_window in time_windows:
        message = f'Calculating {time_window}M features...'
        _logging_info(message)

        df_features_cohort_time_window_list = []
        for cohort in tqdm(cohorts):
            df_features_cohort_time_window = _build_features_customers_for_cohort(
                df_customers, id_customers_col, cohort_customers_col,
                cohort, time_window
            )
            df_features_cohort_time_window_list.append(df_features_cohort_time_window)

        df_features_time_window = pd.concat(df_features_cohort_time_window_list,
                                            axis=0, ignore_index=True)

        df_features = (
            df_features
            .merge(df_features_time_window
                   .rename(columns={'cohort': cohort_audience_col,
                                    id_customers_col: id_audience_col}),
                   on=[id_audience_col,cohort_audience_col],
                   how='left')
            .reset_index(drop=True)
        )

    _check_dataset_granularity(df_features, [id_audience_col,cohort_audience_col],
                               raise_error=True)

    return df_features

def _build_features_geolocation_for_cohort(
        df_geolocation: pd.DataFrame,
        id_geolocation_col: str,
        cohort_geolocation_col: str,
        current_cohort: int,
        time_window: int
)-> pd.DataFrame:

    pre_assign = {}
    agg_functions = {
        f'mean_distance_customer_seller_m{time_window}': ('distance_customer_seller', 'mean'),
        f'std_distance_customer_seller_m{time_window}': ('distance_customer_seller', 'std'),
        f'max_distance_customer_seller_m{time_window}': ('distance_customer_seller', 'max'),
        f'min_distance_customer_seller_m{time_window}': ('distance_customer_seller', 'min'),
    }

    post_assign = {}

    post_drop_cols = []

    df_geolocation_cohort_time_window = (
        _build_features_for_cohort_time_window(df_geolocation,
                                               id_geolocation_col,
                                               cohort_geolocation_col,
                                               current_cohort,
                                               time_window,
                                               pre_assign,
                                               agg_functions,
                                               post_assign,
                                               post_drop_cols)
    )

    return df_geolocation_cohort_time_window

def build_features_geolocation(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_geolocation: pd.DataFrame,
        id_geolocation_col: str,
        cohort_geolocation_col: str,
        time_windows: List[int],
)-> pd.DataFrame:
    """
    """

    df_geolocation = df_geolocation.query('order_status=="delivered"')

    df_features = df_audience[[id_audience_col,cohort_audience_col]]
    cohorts = sorted(df_audience[cohort_audience_col].unique())
    for time_window in time_windows:
        message = f'Calculating {time_window}M features...'
        _logging_info(message)

        df_features_cohort_time_window_list = []
        for cohort in tqdm(cohorts):
            df_features_cohort_time_window = _build_features_geolocation_for_cohort(
                df_geolocation, id_geolocation_col, cohort_geolocation_col,
                cohort, time_window
            )
            df_features_cohort_time_window_list.append(df_features_cohort_time_window)

        df_features_time_window = pd.concat(df_features_cohort_time_window_list,
                                            axis=0, ignore_index=True)

        df_features = (
            df_features
            .merge(df_features_time_window
                   .rename(columns={'cohort': cohort_audience_col,
                                    id_geolocation_col: id_audience_col}),
                   on=[id_audience_col,cohort_audience_col],
                   how='left')
            .reset_index(drop=True)
        )

    _check_dataset_granularity(df_features, [id_audience_col,cohort_audience_col],
                               raise_error=True)

    return df_features

def build_features_sellers(
        df_audience: pd.DataFrame,
        id_audience_col: str,
        cohort_audience_col: str,
        df_sellers: pd.DataFrame,
        id_sellers_col: str)-> pd.DataFrame:
    """
    """

    df_features = (
        df_audience[[id_audience_col,cohort_audience_col]]
        .merge(
            df_sellers[[id_sellers_col, 'seller_state']]
            .rename(columns={id_sellers_col: id_audience_col}),
            on=[id_audience_col],
            how='left'
        )
    )

    return df_features

def _merge_features(
    inference_data: pd.DataFrame,
    features_data: pd.DataFrame,
    prefix: str,
    merge_key: List[str],
)-> pd.DataFrame:
    """
    Realiza o merge da base de features históricas (faturamento ou produto) e renomeia as variáveis
    colocando o prefixo de indentificação.
    """

    renaming_columns_map = {col: prefix+'_'+col
                              for col in features_data.columns
                                if col not in merge_key}
    try:
        _check_dataset_granularity(features_data,
                                   merge_key,
                                   raise_error=True)
    except ConsistenceDataError as exc:
        message = f"The granularity of {prefix} dataset is not {merge_key}."
        raise ConsistenceDataError(message) from exc

    inference_data_merged = (
        inference_data
        .merge(features_data.rename(columns=renaming_columns_map),
               on=merge_key, how="left")
    )

    return inference_data_merged

def _feature_table_final_treatment(inference_data: pd.DataFrame)-> pd.DataFrame:
    """
    Filtra colunas para considerar apenas as variáveis das bases de features. 
    Realiza conversão de tipos e preenchimento de nulos nas colunas.
    """

    inference_data_final = (
        inference_data
        .apply(
            lambda col: (
                col.fillna('NULO').astype(str).astype('category')
                    if col.dtype == 'object' else col.fillna(np.nan)
            )
        )
    )

    return inference_data_final

def build_feature_table(
    df_audience: pd.DataFrame,
    df_feat_orders: pd.DataFrame,
    df_feat_items: pd.DataFrame,
    df_feat_reviews: pd.DataFrame,
    df_feat_payments: pd.DataFrame,
    df_feat_customers: pd.DataFrame,
    df_feat_geolocation: pd.DataFrame,
    df_feat_sellers: pd.DataFrame,
    id_audience_col: str,
    cohort_audience_col: str,
) -> pd.DataFrame:
    """
    """

    inference_data = (
        df_audience
        .pipe(_merge_features, df_feat_orders, "ord", [id_audience_col, cohort_audience_col])
        .pipe(_merge_features, df_feat_items, "itm", [id_audience_col, cohort_audience_col])
        .pipe(_merge_features, df_feat_reviews, "rev", [id_audience_col, cohort_audience_col])
        .pipe(_merge_features, df_feat_payments, "pay", [id_audience_col, cohort_audience_col])
        .pipe(_merge_features, df_feat_customers, "ctm", [id_audience_col, cohort_audience_col])
        .pipe(_merge_features, df_feat_geolocation, "geo", [id_audience_col, cohort_audience_col])
        .pipe(_merge_features, df_feat_sellers, "sel", [id_audience_col, cohort_audience_col])
        .pipe(_feature_table_final_treatment)
        .reset_index(drop=True)
    )

    return inference_data
