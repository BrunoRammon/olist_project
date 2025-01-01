"""
This is a boilerplate pipeline 'scoring'
generated using Kedro 0.19.10
"""

from typing import List
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from olist_project.utils.utils import (
    CustomLGBMClassifier,
    _column_object_to_category,
    _column_numeric_to_float
)

def scoring(
    trained_model: CustomLGBMClassifier,
    inference_data: pd.DataFrame,
    id_col: List[str],
    cohort_col: List[str]
) -> pd.DataFrame:
    """
    Realiza escoragem com modelo treinado a partir da base de features.

    Parametros
    ----------
    trained_model : sklearn.pipeline.Pipeline
        modelo treinado
    inference_data : pandas.DataFrame
        base de features contendo dados de entrada do modelo
    """
    # Check if inference_data is a DataFrame
    if not isinstance(inference_data, pd.DataFrame):
        raise ValueError("inference_data must be a pandas DataFrame")

    if inference_data.empty:
        raise ValueError("inference_data must not be empty")

    # Check if trained_model is a CustomLGBMClassifier
    if not isinstance(trained_model, CustomLGBMClassifier):
        raise ValueError("trained_model must be a CustomLGBMClassifier")

    # Check if model is fitted
    check_is_fitted(trained_model)
    features_in = trained_model.feature_names_in_

    # Check if inference_data has all features
    if not set(features_in).issubset(inference_data.columns):
        raise ValueError("inference_data must have all features")

    X = (
        inference_data[features_in]
        .pipe(_column_object_to_category)
        .pipe(_column_numeric_to_float)
    )
    df_id = inference_data[[id_col,cohort_col]]
    # calculating score
    result = (
        pd.concat([df_id, X], axis=1)
        .assign(
            proba = trained_model.predict_proba(X)[:,1],
            score = trained_model.predict_score(X),
            rating = trained_model.predict_rating(X),
        )
    )

    return result
