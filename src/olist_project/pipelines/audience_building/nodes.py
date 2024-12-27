"""
This is a boilerplate pipeline 'audience_building'
generated using Kedro 0.19.10
"""

from typing import Dict, List, Any
import pandas as pd
from olist_project.utils.utils import (
    _cohort_range_list, _check_dataset_granularity,
    _cohort_offset
)

def build_initial_audience(
        df_orders: pd.DataFrame,
        id_col: str,
        cohort_info_col: str,
        start_cohort: int,
        end_cohort: int,
        historical_period: int
)-> pd.DataFrame:
    """
    """

    df_audience_cohort_list = []
    cohorts = _cohort_range_list(start_cohort,end_cohort)
    for cohort in cohorts:
        initial_cohort = _cohort_offset(cohort, -historical_period)
        df_audience_cohort = (
            df_orders
            .query(f'({cohort_info_col}>={initial_cohort})&({cohort_info_col}<{cohort})')
            .drop_duplicates([id_col])
            .assign(
                cohort = cohort
            )
            [[id_col,'cohort']]
        )
        df_audience_cohort_list.append(df_audience_cohort)

    df_initial_audience = (
        pd.concat(df_audience_cohort_list, axis=0, ignore_index=True)
        .reset_index(drop=True)
    )

    _check_dataset_granularity(df_initial_audience, ['seller_id','cohort'],
                               raise_error=True)

    return  df_initial_audience

def build_audience_filters(
        df_orders: pd.DataFrame,
        id_col: str,
        cohort_info_col: str,
        time_window: int,
)-> pd.DataFrame:
    """
    """

    cohorts = sorted(df_orders[cohort_info_col].unique())
    df_flag_cohort_list = []
    for cohort in cohorts:
        initial_cohort = _cohort_offset(cohort, -time_window)
        df_flag_cohort = (
            df_orders
            .query(f'({cohort_info_col}>={initial_cohort})&({cohort_info_col}<{cohort})')
            .assign(
                flag_delivered_order = lambda df: (df.order_status=="delivered").astype(int)
            )
            .groupby(id_col, as_index=False)
            .agg(**{
                f'flag_has_delivered_order_m{time_window}': ('flag_delivered_order', 'max')
            })
            .assign(
                cohort = cohort
            )
            [[id_col,'cohort', 'flag_has_delivered_order_m9']]
        )
        df_flag_cohort_list.append(df_flag_cohort)

    df_flag_delivered = (
        pd.concat(df_flag_cohort_list, axis=0, ignore_index=True)
        .reset_index(drop=True)
    )

    _check_dataset_granularity(df_flag_delivered, [id_col,'cohort'], raise_error=True)

    return df_flag_delivered

def build_final_audience(
        df_initial_audience: pd.DataFrame,
        df_flag_filters: pd.DataFrame,
        filters_specs: Dict[str, List[Any]],
        id_col: str,
        cohort_col: str,
)-> pd.DataFrame:
    """
    """
    df_final_audience = (
        df_initial_audience
        .merge(
            df_flag_filters,
            on=[id_col,cohort_col],
            how='left'
        )
    )

    for _, flag_filter_spec in filters_specs.items():
        flag_filter_col = flag_filter_spec['filter_column_name']
        select_values = flag_filter_spec['select_values']
        mask = df_final_audience[flag_filter_col].isin(select_values)
        df_final_audience = df_final_audience[mask]

    _check_dataset_granularity(df_final_audience, [id_col,cohort_col], raise_error=True)

    return (
        df_final_audience[[id_col,cohort_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
