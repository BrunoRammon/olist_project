"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.10
"""
from typing import Tuple, List, Dict, Union, Any
import copy
import logging
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.selection import (
    DropDuplicateFeatures, 
    DropConstantFeatures,
    DropHighPSIFeatures,
    SelectByInformationValue,
    SmartCorrelatedSelection,
    RecursiveFeatureAddition,
)
import optuna
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import optbinning
from olist_project.utils.utils import (
    CustomLGBMClassifier,
    _column_object_to_category,
    _column_numeric_to_float,
)
from olist_project.utils.model import get_model, ModelType, objective, MetricType

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

def train_oot_split(
    abt: pd.DataFrame,
    start_cohort: int,
    split_cohort: int,
    final_cohort: int,
    id_col: str,
    cohort_col: str,
    target_col: str,
    features: Union[Dict[str, List[str]], None] = None,
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
    features : Union[Dict[str, List[str]], None]
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

    if not isinstance(id_col, str):
        raise TypeError("id_col must be a list of strings")

    if not isinstance(cohort_col, str):
        raise TypeError("cohort_col must be a list of strings")

    # Check if the id_model_cols are in the abt
    if not all(i in abt.columns for i in [id_col]):
        raise ValueError("id_col must be in the abt columns")

    if not all(i in abt.columns for i in [cohort_col]):
        raise ValueError("cohort_col must be in the abt columns")

    # Check if the cohorts make sense
    if start_cohort >= split_cohort or split_cohort >= final_cohort:
        message = (
            "start_cohort must be less than split_cohort and split_cohort "
            "must be less than final_cohort"
        )
        raise ValueError(message)

    if features:
        col_features = features['final_feature_set']
    else:
        col_features = [
            col for col in abt.columns
                if not col.startswith('target_') and col not in [id_col, cohort_col]
        ]

    # Check if the features are in the abt
    if col_features and not all(i in abt.columns for i in col_features):
        raise ValueError("final feature set must be in the abt columns")

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

    x_train_duplicated_mask = (
        X[mask_train]
        .pipe(_column_object_to_category)
        .pipe(_column_numeric_to_float)
        .duplicated()
    )
    X = X.filter(col_features)
    logger = logging.getLogger(__name__)
    message = (
        f'There are {x_train_duplicated_mask.sum()} repeated rows in the feature train matrix'
        ' that will be dropped.'
    )
    logger.info(message)
    X_train = (
        X[mask_train & (~x_train_duplicated_mask)]
        .pipe(_column_object_to_category)
        .pipe(_column_numeric_to_float)
        .reset_index(drop=True)
    )
    y_train = y[mask_train & (~x_train_duplicated_mask)].reset_index(drop=True)
    id_model_train = id_model[mask_train & (~x_train_duplicated_mask)].reset_index(drop=True)

    X_oot = (
        X[mask_oot]
        .pipe(_column_object_to_category)
        .pipe(_column_numeric_to_float)
        .reset_index(drop=True)
    )
    y_oot = y[mask_oot].reset_index(drop=True)
    id_model_oot = id_model[mask_oot].reset_index(drop=True)

    return (
        X_train, y_train, id_model_train,
        X_oot, y_oot, id_model_oot
    )

def feature_selection(X_train: pd.DataFrame,
                      y_train: pd.DataFrame,
                      target_name: str,
                      nfolds_cv: int):
    """
    """
    y_train = y_train[target_name]
    logger = logging.getLogger(__name__)

    cat_vars = [col for col in X_train.select_dtypes('category').columns]
    num_vars = [col for col in X_train.columns if col not in cat_vars]

    logger.info("Computing feature selection by filter methods...")
    baseline_model = get_model(X_train,cat_vars,num_vars,
                               model_type=ModelType.LGBM)
    pipe_steps = copy.deepcopy(baseline_model.steps)
    pipe_steps.insert(-1,('drop_duplicated', DropDuplicateFeatures()))
    pipe_steps.insert(-1,('num_imputer', ArbitraryNumberImputer(-99999999)))
    pipe_steps.insert(-1,('drop_constant', DropConstantFeatures(tol=.95)))
    pipe_steps.insert(-1,('drop_psi', DropHighPSIFeatures(threshold=0.1)))
    feature_psi_pipe = Pipeline(pipe_steps)
    X_train_trans_psi = feature_psi_pipe[:-1].fit_transform(X_train,y_train)

    features_iv = [col for col in X_train_trans_psi.columns if col not in ['sel_seller_state']]
    iv_selection = SelectByInformationValue(variables=features_iv,
                                            strategy='equal_frequency',
                                            threshold=0.2)
    iv_selection.fit(X_train_trans_psi,y_train)
    selected_features_filter_methods = iv_selection.get_feature_names_out()
    message = (
        "End of feature selection by filter methods."
        f"\n  {len(selected_features_filter_methods)} features selected."
    )
    logger.info(message)

    logger.info("Computing feature selection by ML based methods...")

    skf = StratifiedKFold(nfolds_cv)

    # smart correlated selection
    pipe_steps = copy.deepcopy(baseline_model.steps)
    pipe_steps.insert(-1,
        ('drop_correlated', SmartCorrelatedSelection(selection_method='model_performance',
                                                     estimator=baseline_model[-1],
                                                     cv=skf))
    )
    corr_feature_sel_ml_based_pipe = Pipeline(pipe_steps)
    corr_feature_sel_ml_based_pipe[:-1].fit(X_train[selected_features_filter_methods],
                                            y_train)
    corr_feature_set = (
        corr_feature_sel_ml_based_pipe
        .named_steps['drop_correlated']
        .get_feature_names_out()
    )
    message = (
        "End of smart correlated feature selection. "
        f"{len(corr_feature_set)} features selected."
    )
    logger.info(message)

    # RFA selection
    cat_vars = [col for col in (X_train[corr_feature_set]
                                .select_dtypes('category')
                                .columns)]
    num_vars = [col for col in X_train[corr_feature_set].columns
                    if col not in cat_vars]
    baseline_model = get_model(X_train[corr_feature_set],cat_vars,num_vars,
                               model_type=ModelType.LGBM)
    pipe_steps = copy.deepcopy(baseline_model.steps)
    threshold_rfa = 0.001
    pipe_steps.insert(-1,
        ('rfa_selection', RecursiveFeatureAddition(estimator=baseline_model[-1],
                                                   cv=skf,
                                                   threshold=threshold_rfa))
    )
    rfa_feature_sel_ml_based_pipe = Pipeline(pipe_steps)
    rfa_feature_sel_ml_based_pipe[:-1].fit(X_train[corr_feature_set], y_train)
    final_feature_set = (
        rfa_feature_sel_ml_based_pipe
        .named_steps['rfa_selection']
        .get_feature_names_out()
    )

    message = (
        "End of RFA feature selection. "
        f"{len(final_feature_set)} features selected."
        "\nEnd of feature selection by ML methods."
    )
    logger.info(message)

    return {'filter_methods_feature_set': selected_features_filter_methods,
            'smart_corr_feature_set': corr_feature_set,
            'final_feature_set': final_feature_set}

def hyperparameters_tuning(X_train: pd.DataFrame,
                           y_train: pd.DataFrame,
                           n_trials: int,
                           target_name: str,
                           random_state: int,
                           nfolds_cv: int):
    """
    """
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(study_name='optmization_study',
                                directions=['maximize'],
                                sampler=sampler)
    y_train = y_train[target_name]
    base_model_type = ModelType.LGBM
    validation_type = MetricType.TRAIN_TEST_CV_PREDICT
    get_model_func = get_model
    study.optimize(lambda trial: objective(trial,
                                           X_train, y_train,
                                           validation_type=validation_type,
                                           with_std_penalization=True,
                                           feature_selection=False,
                                           model_type=base_model_type,
                                           get_model_function=get_model_func,
                                           cv_n_folds=nfolds_cv),
                   n_trials=n_trials)
    model_best_hyperparams = study.best_trial.params
    return model_best_hyperparams

def _calculate_probas(model,X_dev,y_dev, nfolds_cv):

    skf = StratifiedKFold(nfolds_cv)
    y_proba_oos = cross_val_predict(model, X_dev, y_dev,
                                    method='predict_proba', cv=skf)[:,1]
    y_proba_oos = pd.Series(y_proba_oos, index=X_dev.index, name='proba')
    return y_proba_oos

def _calulate_scores(y_true, y_proba,
                     random_state):

    y_proba_df = pd.DataFrame({'proba': y_proba},
                               index=y_true.index)
    quant_transf = QuantileTransformer(random_state=random_state).set_output(transform='pandas')
    quant_transf.fit(y_proba_df)
    score = (1-pd.Series(quant_transf.transform(y_proba_df).proba,
                               index=y_true.index, name='score'))*1000

    return score

def _get_discretizers(y_true,score,monotonic_trend='auto',
                     n_ratings=5,max_n_bins=None,min_n_bins=None,
                     min_event_rate_diff=0,
                     prebinning_method='cart', solver='cp', divergence='iv',
                     max_n_prebins=20, min_prebin_size=0.05, min_bin_size=None,
                     max_bin_size=None, min_bin_n_nonevent=None,
                     max_bin_n_nonevent=None, min_bin_n_event=None,
                     max_bin_n_event=None, max_pvalue=None,
                     max_pvalue_policy='consecutive', gamma=0,
                     outlier_detector=None, outlier_params=None, class_weight=None,
                     cat_cutoff=None, cat_unknown=None, user_splits=None,
                     user_splits_fixed=None, special_codes=None, split_digits=None,
                     mip_solver='bop', time_limit=100, verbose=False):
    if n_ratings is not None:
        max_n_bins = n_ratings
        min_n_bins = n_ratings
    disc_opt = optbinning.OptimalBinning(max_n_bins=max_n_bins,
                                         min_n_bins=min_n_bins,
                                         monotonic_trend=monotonic_trend,
                                         min_event_rate_diff=min_event_rate_diff,
                                         prebinning_method=prebinning_method,
                                         solver=solver,
                                         divergence=divergence,
                                         max_n_prebins=max_n_prebins,
                                         min_prebin_size=min_prebin_size,
                                         min_bin_size=min_bin_size,
                                         max_bin_size=max_bin_size,
                                         min_bin_n_nonevent=min_bin_n_nonevent,
                                         max_bin_n_nonevent=max_bin_n_nonevent,
                                         min_bin_n_event=min_bin_n_event,
                                         max_bin_n_event=max_bin_n_event,
                                         max_pvalue=max_pvalue,
                                         max_pvalue_policy=max_pvalue_policy,
                                         gamma=gamma,
                                         outlier_detector=outlier_detector,
                                         outlier_params=outlier_params,
                                         class_weight=class_weight,
                                         cat_cutoff=cat_cutoff,
                                         cat_unknown=cat_unknown,
                                         user_splits=user_splits,
                                         user_splits_fixed=user_splits_fixed,
                                         special_codes=special_codes,
                                         split_digits=split_digits,
                                         mip_solver=mip_solver,
                                         time_limit=time_limit,
                                         verbose=verbose)
    disc_opt.fit(score, y_true)
    faixas_opt = disc_opt.splits.tolist()
    return faixas_opt

def ratings_optimization(X_train: pd.DataFrame,
                         y_train: pd.DataFrame,
                         best_hyperparameters: Dict[str, Any],
                         nratings: int,
                         target_name: str,
                         random_state: int,
                         nfolds_cv: int):
    """
    """
    cat_vars = [col for col in (X_train.select_dtypes('category').columns)]
    num_vars = [col for col in X_train.columns
                    if col not in cat_vars]
    model = get_model(X_train,cat_vars,num_vars,
                      model_type=ModelType.LGBM,
                      params=best_hyperparameters)
    y_train = y_train[target_name]
    y_proba_oos = _calculate_probas(model,X_train,y_train,nfolds_cv)
    score_oos = _calulate_scores(y_train,y_proba_oos,random_state)
    ratings_limits = _get_discretizers(y_train,score_oos, n_ratings=nratings,
                                       monotonic_trend='descending')
    return ratings_limits

def train_final_model(X_train: pd.DataFrame,
                      y_train: pd.DataFrame,
                      hyperparams: Dict[str, Any],
                      ratings: Union[List[float], Dict[str,float]],
                      random_state: int,
                      nfolds_cv: Union[int,None] = None)-> CustomLGBMClassifier:
    """
    """
    model = CustomLGBMClassifier(random_state=random_state,
                                 **hyperparams)
    target_name = y_train.columns[0]
    model.fit_all(X_train, y_train[target_name], nfolds_cv=nfolds_cv, ratings=ratings)
    return model

def _create_df_results(
    trained_model: CustomLGBMClassifier,
    X: pd.DataFrame,
    y: pd.DataFrame,
    id_model: pd.DataFrame
)-> pd.DataFrame:
    """
    Calcula os resultados do modelo dados conjuntos de X e y.
    """

    df_result = (
        pd.concat([id_model, X, y], axis=1)
        .assign(
            score = (
                trained_model.predict_score(X)
            ),
            rating = (
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
    X_oot: pd.DataFrame,
    y_oot: pd.DataFrame,
    id_model_oot: pd.DataFrame,
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

    check_is_fitted(trained_model)
    oot_results = _create_df_results(trained_model, X_oot, y_oot,
                                     id_model_oot)
    train_results = _create_df_results(trained_model, X_train, y_train,
                                       id_model_train)

    return train_results, oot_results
