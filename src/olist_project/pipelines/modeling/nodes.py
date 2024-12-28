"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""
from typing import Tuple, List, Dict, Union
import pandas as pd
from sklearn.utils.validation import check_is_fitted
import logging
from olist_project.utils.utils import (
    CustomLGBMClassifier
)

def _info_merge(raw_spine: pd.DataFrame, features: pd.DataFrame,
                id_col: str, cohort_col: str):
    """
    """

    safras = list(set(features[cohort_col]).intersection(set(raw_spine[cohort_col])))
    raw_spine_filter = raw_spine.query(f"{cohort_col}.isin({safras})")
    features_filter = features.query(f"{cohort_col}.isin({safras})")

    df_merge = raw_spine_filter.merge(features_filter, on=[id_col, cohort_col],
                                      how='outer', indicator=True)
    val_counts = df_merge['_merge'].value_counts()
    abt_lenght = val_counts.loc['both']
    only_spine = val_counts.loc['left_only']
    only_features = val_counts.loc['right_only']

    message = (
        f"Existem {only_spine} clientes no público da base da target "
        "que não estão no público da base de features.\n"
        f"Existem {only_features} clientes no público da base de features "
        "que não estão no público da base da target.\n"
        f"Existem {abt_lenght} clientes que estão nos públicos das duas bases e, " 
        "por isso, serão considerados na construção da ABT."
    )
    logger = logging.getLogger(__name__)
    logger.info(message)

    return None

def build_abt(raw_spine: pd.DataFrame,
              features: pd.DataFrame,
              id_col: str,
              cohort_col: str,
              drop_columns: Union[List[str], None] = None,)-> pd.DataFrame:
    """
    Constroi a ABT a partir de junçã da tabela spine (com target) com tabela de features.

    Parametros
    ----------
    raw_spine : pandas.DataFrame
        Tabela spine contentendo as variáveis reposta do modelo.
    features : pandas.DataFrame
        Tabela de features, contendo as variáveis preditivas do modelo.
    """

    if drop_columns is None:
        drop_columns = []

    _info_merge(raw_spine, features, id_col, cohort_col)
    abt = (
        raw_spine
        .merge(features, on=[id_col, cohort_col], how='inner')
        .reset_index(drop=True)
        .drop(columns=drop_columns)
    )
    return abt


def _column_object_to_category(df: pd.DataFrame)-> pd.DataFrame:
    """
    """

    return df.apply(lambda col: col.astype('category') if col.dtype in ['object','string'] else col)

def train_oot_split(
    abt: pd.DataFrame,
    start_cohort: int,
    split_cohort: int,
    final_cohort: int,
    id_col: str,
    cohort_col: str,
    target_col: str,
    features: Union[List[str], None] = None,
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide a ABT nos datasets de treino e validação oot e retorna a matriz de features, o vetor
    da variável resposta e um as colunas que identificam unicamente cada observação (CNPJ_CPF e 
    SAFRA).
    
    Parametros
    ----------
    abt : pandas.DataFrame
        Dataframe da ABT do modelo.
    start_cohort : int
        Safra inicial, sendo um inteiro no formato YYYYmm
    split_cohort : int
        Safra que divide período de treino do período de validação do modelo, sendo um inteiro
        no formato YYYYmm. Essa safra é incluída na validação mas excluída do treino.
    final_cohort : int
        Safra final utilizada na modelagem.
    id_model_cols : List[str]
        Lista de nome das colunas que identificam unicamente as observações usadas na modelagem.
    target_col : str
        Nome da coluna da target usada na modelagem. Por exemplo, TARGET_FLX
    features : List[str]
        Lista de nome das colunas das variáveis utilizadas no modelo.
    """

    # Check if the inputs are dataframes
    if not isinstance(abt, pd.DataFrame):
        raise TypeError("abt must be a pandas.DataFrame")

    # Check if the DataFrames are not empty
    if abt.empty:
        raise ValueError("abt must not be empty")

    # Check if the inputs are integers
    if (not isinstance(start_cohort, int) or
        not isinstance(split_cohort, int) or
        not isinstance(final_cohort, int)):
        message = (
            "start_cohort, split_cohort and final_cohort must be integers"
        )
        raise TypeError(message)

    # Check if the target_col is a string and is in the abt
    if not isinstance(target_col, str):
        raise TypeError("target_col must be a string")

    if target_col not in abt.columns:
        raise ValueError("target_col must be in the abt columns")

    # Check if the inputs are lists of strings
    if features and not all(isinstance(i, str) for i in features):
        raise TypeError("features must be a list of strings")

    if not isinstance(id_col, str):
        raise TypeError("id_col must be a list of strings")

    if not isinstance(cohort_col, str):
        raise TypeError("cohort_col must be a list of strings")

    # Check if the id_model_cols are in the abt
    if not all(i in abt.columns for i in [id_col]):
        raise ValueError("id_col must be in the abt columns")

    if not all(i in abt.columns for i in [cohort_col]):
        raise ValueError("cohort_col must be in the abt columns")

    # Check if the features are in the abt
    if features and not all(i in abt.columns for i in features):
        raise ValueError("features must be in the abt columns")

    # Check if the cohorts make sense
    if start_cohort >= split_cohort or split_cohort >= final_cohort:
        message = (
            "start_cohort must be less than split_cohort and split_cohort "
            "must be less than final_cohort"
        )
        raise ValueError(message)

    if features:
        col_features = features
    else:
        col_features = [
            col for col in abt.columns
                if not col.startswith('target_') and col not in [id_col, cohort_col]
        ]

    all_features = [
        col for col in abt.columns
            if not col.startswith('target_') and col not in [id_col, cohort_col]
    ]
    X = abt.filter(all_features)
    map_dtypes = {col: float for col in X.select_dtypes('int').columns}
    X = X.astype(map_dtypes)
    y = abt.filter([target_col]).reset_index(drop=True)
    id_model = abt.filter([id_col, cohort_col]).reset_index(drop=True)
    X = X.reset_index(drop=True)

    mask_train = (id_model[cohort_col] >= start_cohort) & (id_model[cohort_col] <= split_cohort)
    mask_oot = (id_model[cohort_col] > split_cohort) & (id_model[cohort_col] <= final_cohort)

    x_train_duplicated_mask = _column_object_to_category(X[mask_train]).duplicated()
    X = X.filter(col_features)
    logger = logging.getLogger(__name__)
    message = (
        f'There are {x_train_duplicated_mask.sum()} repeated rows in the feature train matrix'
        ' that will be dropped.'
    )
    logger.info(message)
    X_train = (
        _column_object_to_category(X[mask_train & (~x_train_duplicated_mask)])
        .reset_index(drop=True)
    )
    y_train = y[mask_train & (~x_train_duplicated_mask)].reset_index(drop=True)
    id_model_train = id_model[mask_train & (~x_train_duplicated_mask)].reset_index(drop=True)

    X_oot = _column_object_to_category(X[mask_oot].reset_index(drop=True))
    y_oot = y[mask_oot].reset_index(drop=True)
    id_model_oot = id_model[mask_oot].reset_index(drop=True)

    return (
        X_train, y_train, id_model_train,
        X_oot, y_oot, id_model_oot
    )

def train_final_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
                      ratings: Union[List[float], Dict[str,float]],
                      random_state: int, hyperparams: Dict)-> CustomLGBMClassifier:
    """
    """
    model = CustomLGBMClassifier(random_state=random_state,
                                 **hyperparams)
    model.fit_all(X_train, y_train, ratings=ratings)
    return model

def _create_df_results(
    trained_model: CustomLGBMClassifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
    id_model: pd.DataFrame,
    groups: pd.Series
)-> pd.DataFrame:
    """
    Calcula os resultados do modelo dados conjuntos de X e y.
    """

    df_result = (
        pd.concat([id_model, X, y, groups], axis=1)
        .assign(
            PROBA = (
                trained_model.predict_proba(X)[:,1]
            ),
            SCORE = (
                trained_model.predict_score(X, transform_score_by_rating=True)
            ),
            RATING = (
                trained_model.predict_rating(X)
            )
        )
    )

    return  df_result

def model_results(
    trained_model: CustomLGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    id_model_train: pd.DataFrame,
    df_tp_neg_train: pd.DataFrame,
    X_oot: pd.DataFrame,
    y_oot: pd.DataFrame,
    id_model_oot: pd.DataFrame,
    df_tp_neg_oot: pd.DataFrame,
)-> pd.DataFrame:
    """
    Cria a tabela de resultado de treino e validação para o modelo treinado.

    Parametros
    ----------
    X_train : pandas.DataFrame
        Conjunto features de treinamento.
    y_train : pandas.DataFrame
        Vetor de treino da variável resposta.
    id_model_train : pandas.DataFrame
        Identificador único das observações para o conjunto de treino.
    X_oot : pandas.DataFrame
        Conjunto features de validação oot.
    y_oot : pandas.DataFrame
        Vetor de validação oot da variável resposta.
    id_model_oot : pandas.DataFrame
        Identificador único das observações para o conjunto de validação oot.
    n_folds : int
        Número de folds usado no cálculo dos resultado para o conjunto de treino.
    """
    # Assuming the code is inside _create_df_results or just before calling it

    if (not (len(X_train) == len(y_train) == len(id_model_train)) or
        not (len(X_oot) == len(y_oot) == len(id_model_oot))):
        raise ValueError("Input data frames do not have matching lengths.")

    check_is_fitted(trained_model)
    oot_results = _create_df_results(trained_model, X_oot, y_oot,
                                     id_model_oot,
                                     df_tp_neg_oot['CAD_TIPO_NEGOCIO'])
    train_results = _create_df_results(trained_model, X_train, y_train,
                                       id_model_train,
                                       df_tp_neg_train['CAD_TIPO_NEGOCIO'])

    return train_results, oot_results
