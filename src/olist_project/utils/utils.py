"""
Functions used in pipelines scripts
"""

from typing import List, Dict, Union
import pandas as pd
import numpy as np
import datetime as dt
from lightgbm import LGBMClassifier
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2
import string
from sklearn.preprocessing import QuantileTransformer

def _map_tn(x):

    check_tn = any(x==tp_neg for tp_neg in ['POSTO IPIRANGA TERCEIROS', 'POSTO LOCADO',
                                            'POSTO BANDEIRA BRANCA', 'POSTO IPIRANGA PROPRIO'])
    return x if check_tn else 'OUTROS'

def _cohort_to_datetime(cohort: Union[pd.Series,int]):
    """
    """
    if isinstance(cohort, pd.Series):
        return pd.to_datetime(cohort.astype('string'),format='%Y%m')
    elif isinstance(cohort, int) or isinstance(cohort, np.int64):
        return pd.to_datetime(str(cohort),format='%Y%m')
    else:
        raise TypeError()

def _datetime_to_cohort(dt_cohort: Union[pd.Series,pd.Timestamp]):
    """
    """
    if isinstance(dt_cohort, pd.Series):
        return dt_cohort.dt.strftime("%Y%m").astype('Int64')
    elif isinstance(dt_cohort, pd.Timestamp):
        return int(dt_cohort.strftime("%Y%m"))
    else:
        raise TypeError()

def _cohort_offset(cohort: Union[pd.Series, int], cohort_offset: int):
    """
    """

    dt_cohort = _cohort_to_datetime(cohort)
    dt_cohort_offset = dt_cohort + pd.DateOffset(months=cohort_offset)

    return _datetime_to_cohort(dt_cohort_offset)

def _months_between_cohort(cohort_1: pd.Series,cohort_2, freq='M'):
    dt_cohort_1 = _cohort_to_datetime(cohort_1)
    dt_cohort_2 = _cohort_to_datetime(cohort_2)
    return (dt_cohort_2.dt.to_period(freq).astype(int) - dt_cohort_1.dt.to_period(freq).astype(int))

def _cohort_range_list(start_cohort, end_cohort, freq='MS'):
    start_dt_cohort = _cohort_to_datetime(start_cohort)
    end_dt_cohort = _cohort_to_datetime(end_cohort)
    return [
        _datetime_to_cohort(dt_cohort)
            for dt_cohort in list(pd.date_range(start=start_dt_cohort,end=end_dt_cohort,freq=freq))
    ]

def _cohort_series_to_datetime(cohort: pd.Series):
    """
    """
    return pd.to_datetime(cohort.astype('string'), format='%Y%m')

def _cohort_subtraction(cohort_1: pd.Series, cohort_2: pd.Series):
    """
    """

    dt_cohort_1 = _cohort_series_to_datetime(cohort_1)
    dt_cohort_2 = _cohort_series_to_datetime(cohort_2)

    return (dt_cohort_1-dt_cohort_2).dt.days

def _check_length_cnpj_cpf(cnpj_cpf: pd.Series,
                           cnpj_lenght: int,
                           cpf_lenght: int)-> bool:
    """
    Realiza checagem de número de dígitos de CNPJ_CPF
    """
    # Check if cnpj_cpf is in the correct format (int-castable string)
    if not cnpj_cpf.str.isnumeric().all():
        raise ConsistenceDataError(
            "CNPJ_CPF must be a numeric string in table. "+
            "Please check your data."
        )

    # Check if cnpj_cpf is a Series
    if cnpj_cpf.isna().any():
        raise ConsistenceDataError(
            "CNPJ_CPF cannot have missing values in table. "+
            "Please check your data."
        )

    len_cnpj_cpf = cnpj_cpf.str.len()

    # Check if cnpj_cpf has the correct number of digits
    checking = ((len_cnpj_cpf == cnpj_lenght) | (len_cnpj_cpf == cpf_lenght)).all()
    if not checking:
        raise ValueError(
            f"CNPJ_CPF must have {cnpj_lenght} or {cpf_lenght} digits in "+
            "table. Please check your data."
        )

    return checking

def _drop_check_digits(cnpj_cpf: pd.Series)-> pd.Series:
    """ 
    Retorna CNPJ_CPF raiz, sem os dois últimos dígitos, que é a granularidade do modelo.
    """
    _check_length_cnpj_cpf(cnpj_cpf,14,11)
    return cnpj_cpf.str.slice(0, -2)


def _check_dataset_granularity(
    df: pd.DataFrame,
    colunas: List[str],
    raise_error: bool = False
)-> bool:
    """
    Checa se a combinação de colunas descreve a granularidade da base, isto é,
    não há uma combinação de valores dessas colunas que se repete na base.
    """

    check = df[colunas].value_counts().max() == 1
    if raise_error and not check:
        message_error = f"Existem combinações de valores de {colunas} que se repetem."
        raise ConsistenceDataError(message_error)

    return check

def _date_serial_number(serial_number: int) -> dt.datetime:
    """
    Convert an Excel serial number to a Python datetime object
    :param serial_number: the date serial number
    :return: a datetime object
    """
    # Excel stores dates as "number of days since 1900"

    delta = dt.datetime(1899, 12, 30) + dt.timedelta(days=serial_number)
    return delta

def _float_to_datetime(series_float_date: Union[pd.Series, List],
                       tz: Union[str, None]='UTC') -> dt.datetime:
    """
    Função para converter colunas float para timestamp (tz aware). 
    É padrão de bases que foram extraídas a partir de arquivos excel.
    """
    delta = (
        dt.datetime(1899, 12, 30) +
        pd.to_timedelta(series_float_date, unit='d')
    ).dt.tz_localize(tz)

    return delta

def _convert_column_to_datetime_tz_aware(serie: pd.Series, tz='UTC'):
    """
    converting date columns to datetime64[ns, UTC] dtype to conform data
    """

    if not isinstance(serie.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        if isinstance(serie.dtype, np.dtypes.Float64DType): # pylint: disable=E1101
            return _float_to_datetime(serie, tz)
        if isinstance(serie.dtype, pd.core.arrays.integer.Int64Dtype): # pylint: disable=E1101
            return _float_to_datetime(serie, tz)
        if isinstance(serie.dtype, np.dtypes.DateTime64DType): # pylint: disable=E1101
            return serie.dt.tz_localize(tz)
        if isinstance(serie.dtype, np.dtypes.ObjectDType): # pylint: disable=E1101
            return _float_to_datetime(serie.astype(float), tz)
        else:
            s_dtype = serie.dtype
            message_error = f"A conversão para o tipo de dado {s_dtype} não foi implementada."
            raise NotImplementedError(message_error)

    return serie

def _compare_series(series1, series2, series3=None, labels=None, title=None):
    s1 = set(series1)
    s2 = set(series2)

    if labels is None:
        labels = ['Series 1', 'Series 2', 'Series 3']

    if series3 is None:
        union = s1 | s2
        intersection = s1 & s2

        c1, c2 = ('#0072B2', '#D55E00') 
        label_color = {labels[0]: c1, labels[1]: c2} 

        plt.figure(figsize=(4, 4))
        v = venn2([s1, s2], set_colors=(c1, c2), set_labels=(labels[0], labels[1]))

        for idx, label in enumerate(v.set_labels):
            label.set_color(label_color[label.get_text()])

        plt.title(title)
        plt.show()

        print(f"Total de valores únicos nas duas bases: {len(union)}\n")
        print(f"Total de valores únicos em cada uma das bases:\n{labels[0]}: {len(s1)}\n{labels[1]}: {len(s2)}\n")
        print(f"Valores presentes nas duas bases: {len(intersection)} ({round(len(intersection) / len(union) * 100, 2)}%)")
        print(f"Valores presentes apenas na {labels[0]}: {len(s1 - s2)} ({round(len(s1 - s2) / len(union) * 100, 2)}%)")
        print(f"Valores presentes apenas na {labels[1]}: {len(s2 - s1)} ({round(len(s2 - s1) / len(union) * 100, 2)}%)")
    
    else:
        s3 = set(series3)
        in_all = len(s1 & s2 & s3)
        in_2_of_3 = len((s1 & s2) | (s1 & s3) | (s2 & s3)) - in_all
        only_in_series1 = len(s1 - s2 - s3)
        only_in_series2 = len(s2 - s1 - s3)
        only_in_series3 = len(s3 - s1 - s2)

        c1, c2, c3 = ('#0072B2', '#D55E00', '#009E73')  
        label_color = {labels[0]: c1, labels[1]: c2, labels[2]: c3}

        plt.figure(figsize=(6, 6))
        v = venn3([s1, s2, s3], set_colors=(c1, c2, c3), set_labels=(labels[0], labels[1], labels[2]), )

        for idx, label in enumerate(v.set_labels):
            label.set_color(label_color[label.get_text()])

        plt.show()

        print(f"Total de valores únicos nas três bases: {len(s1 | s2 | s3)}")
        print(f"Total de valores únicos em cada uma das bases:\n{labels[0]}: {len(s1)}\n{labels[1]}: {len(s2)}\n{labels[2]}: {len(s3)}\n")
        print(f"Valores presentes nas três bases: {in_all} ({round(in_all / len(s1 | s2 | s3) * 100, 2)}%)")
        print(f"Valores presentes em duas das três bases: {in_2_of_3} ({round(in_2_of_3 / len(s1 | s2 | s3) * 100, 2)}%)")
        print(f"Valores presentes apenas na {labels[0]}: {only_in_series1} ({round(only_in_series1 / len(s1 | s2 | s3) * 100, 2)}%)")
        print(f"Valores presentes apenas na {labels[1]}: {only_in_series2} ({round(only_in_series2 / len(s1 | s2 | s3) * 100, 2)}%)")
        print(f"Valores presentes apenas na {labels[2]}: {only_in_series3} ({round(only_in_series3 / len(s1 | s2 | s3) * 100, 2)}%)")

class ConsistenceDataError(Exception):
    "Raised when a data checking is not passed"

class CustomLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    Classe para criar um classificador customizado. O classificador é um sklearn pipeline
    composto de duas etapas: um encoder (RareLabelEncoder) e um estimador (LGBMClassifier).
    Essa classe contém métodos específico de problema de crédito de prever scores de crédito
    (predict_score) e ratings (predict_rating). Além disso, trata do problema do LGBMClassifier
    não permitir uso de dataframes com colunas cujos nomes contenham o caracter ':'.
    """
    def __init__(self,
                 boosting_type = 'gbdt',
                 class_weight = None,
                 colsample_bytree = 0.4569936912978631,
                 importance_type = 'split',
                 learning_rate = 0.0080831519139021,
                 max_depth = 2,
                 min_child_samples = 86,
                 min_child_weight = 0.001,
                 min_split_gain = 0.0,
                 n_estimators = 474,
                 n_jobs = None,
                 num_leaves = 17,
                 objective = None,
                 random_state = 42,
                 reg_alpha = 0.026077933417206954,
                 reg_lambda = 0.0016198998041605582,
                 subsample = 0.44493477315950647,
                 subsample_for_bin = 200000,
                 subsample_freq = 3,
                 verbosity = -1,
                 is_unbalance = False):
        self.boosting_type = boosting_type
        self.class_weight = class_weight
        self.colsample_bytree = colsample_bytree
        self.importance_type = importance_type
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.num_leaves = num_leaves
        self.objective = objective
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.subsample_for_bin = subsample_for_bin
        self.subsample_freq = subsample_freq
        self.verbosity = verbosity
        self.is_unbalance = is_unbalance

        self.estimator = LGBMClassifier(
            boosting_type = self.boosting_type,
            class_weight = self.class_weight,
            colsample_bytree = self.colsample_bytree,
            importance_type = self.importance_type,
            learning_rate = self.learning_rate,
            max_depth = self.max_depth,
            min_child_samples = self.min_child_samples,
            min_child_weight = self.min_child_weight,
            min_split_gain = self.min_split_gain,
            n_estimators = self.n_estimators,
            n_jobs = self.n_jobs,
            num_leaves = self.num_leaves,
            objective = self.objective,
            random_state = self.random_state,
            reg_alpha = self.reg_alpha,
            reg_lambda = self.reg_lambda,
            subsample = self.subsample,
            subsample_for_bin = self.subsample_for_bin,
            subsample_freq = self.subsample_freq,
            verbosity = self.verbosity,
            is_unbalance = self.is_unbalance,
        )
        self.classifier = None
        self.is_fitted_ = False

        self.ratings_is_fitted_ = False
        self.score_by_groups_ = False
        self.score_is_fitted_ = False
        self.n_ratings = dict()
        self.score_limits = dict()
        self.ratings_labels = dict()
        self.feature_names_in_ = None
        self.proba_quantile_transformer = (
            QuantileTransformer(random_state=self.random_state)
            .set_output(transform='pandas')
        )
        self.proba_quantile_transformers_ = dict()
    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted_ and self.ratings_is_fitted_

    @property
    def classes_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_classes]: The class label array."""

        if not self.is_fitted_:
            raise NotFittedError('No classes found. Need to call fit beforehand.')
        return self.classifier.classes_  # type: ignore[return-value]

    def __check_unique_groups(self, unique_groups: List[str]):
        groups_fitted = self.proba_quantile_transformers_.keys()
        if len(unique_groups) != len(groups_fitted):
            message = (
                "There must be the same number of categories as seen during the score fit."
            )
            raise ValueError(message)
        if any([group not in groups_fitted for group in unique_groups]):
            message = (
                "All unique groups must be the same as those seen during the score fit."
            )
            raise ValueError(message)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Treina o modelo
        """

        pipe_steps = [
            ('cat_imputer',CategoricalImputer()),
            ('rare_encoder',RareLabelEncoder()),
            ('estimator',self.estimator)
        ]
        self.classifier = Pipeline(pipe_steps)
        self.classifier.fit(X, y)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.is_fitted_ = True

    def predict_proba(self, X: pd.DataFrame):
        """
        Prever as probabilidades do modelo.
        """
        return self.classifier.predict_proba(X)

    def transform(self, X: pd.DataFrame):
        """
        Realiza as estapas de transformação da dataframe passado como argumento.
        """
        if hasattr(self.classifier[:-1],'transform'):
            return  self.classifier[:-1].transform(X)
        else:
            return X

    def fit_score(self, X: pd.DataFrame,
                  y: Union[pd.Series, None]=None,
                  nfolds_cv: Union[None, int]=None,
                  groups: Union[None,pd.Series] = None):
        if isinstance(nfolds_cv,int) and y is not None:
            strat_kfold = StratifiedKFold(nfolds_cv)
            y_proba = cross_val_predict(self.classifier, X, y,
                                        method='predict_proba',
                                        cv=strat_kfold)[:,1]
        else:
            y_proba = self.classifier.predict_proba(X)[:,1]
        y_proba_df = pd.DataFrame({'proba': y_proba})
        if groups is not None:
            for group in groups.unique():
                mask = groups==group
                quant_trans = (
                    QuantileTransformer(random_state=self.random_state)
                    .set_output(transform='pandas')
                )
                self.proba_quantile_transformers_[group] = quant_trans.fit(y_proba_df[mask])
            self.score_by_groups_ = True
        else:
            quant_trans = (
                QuantileTransformer(random_state=self.random_state)
                .set_output(transform='pandas')
            )
            self.proba_quantile_transformers_['unique_group'] = quant_trans.fit(y_proba_df)
        self.score_is_fitted_ = True

    @staticmethod
    def _get_rating_labels(n_labels):
        return list(string.ascii_uppercase[:n_labels][::-1])

    def fit_ratings(self, ratings: Union[List[float], Dict[str,List[float]]]):
        """
        Define os ratings
        """

        if isinstance(ratings,list) and not self.score_by_groups_:
            self.n_ratings['unique_group'] = len(ratings)+1
            self.score_limits['unique_group'] = [-float('inf')] + ratings + [float('inf')]
            self.ratings_labels['unique_group'] = list(self._get_rating_labels(self.n_ratings['unique_group']))
        elif isinstance(ratings,dict) and not self.score_by_groups_:
            self.n_ratings['unique_group'] = len(ratings.keys())+1
            self.score_limits['unique_group'] = [-float('inf')] + list(ratings.values())[:-1] + [float('inf')]
            self.ratings_labels['unique_group'] = list(ratings.keys())
        elif isinstance(ratings,dict) and self.score_by_groups_:
            try:
                self.__check_unique_groups(list(ratings.keys()))
            except ValueError as exc:
                message = (
                    "The dictionary keys must have the same categories"
                    " as the categories of groups seen during the score fit."
                )
                raise Exception(message).with_traceback(exc.__traceback__)

            for group in list(self.proba_quantile_transformers_.keys()):
                ratings_group = ratings[group]
                self.n_ratings[group] = len(ratings_group)+1
                self.score_limits[group] = (
                    [-float('inf')] +
                    ratings_group +
                    [float('inf')]
                )
                self.ratings_labels[group] = list(
                    string.ascii_uppercase[:self.n_ratings[group]][::-1]
                )
        elif isinstance(ratings,list) and self.score_by_groups_:
            message = (
                "If score was fitted by group then the ratings must be a dict"
            )
            raise TypeError(message)

        self.ratings_is_fitted_ = True

    def fit_all(self, X: pd.DataFrame, y: pd.Series,
                groups: Union[None,pd.Series] = None,
                nfolds_cv: Union[int, None] = None,
                ratings: Union[List[float], Dict[str,float]] = None):
        self.fit(X, y)
        self.fit_score(X, y, nfolds_cv, groups=groups)
        self.fit_ratings(ratings)

    def predict_default_probability(self, X: pd.DataFrame):
        """
        Retorna a previsão das probabilidades de inadimplência.
        """

        check_is_fitted(self.classifier)
        return self.predict_proba(X)[:, 1]

    def predict_score(self, X: pd.DataFrame, groups: Union[None, pd.Series]=None,
                      transform_score_by_rating=False):
        """
        Retorna a previsão dos scores.
        """

        check_is_fitted(self.classifier)

        if not self.score_is_fitted_:
            message = (
                "The score must be fitted to use this method. Use fit_score method " 
                "to fit the scores of this classifier."
            )
            raise NotFittedError(message)

        if transform_score_by_rating and not self.ratings_is_fitted_:
            message = (
                "The ratings must be fitted to use this method with the option" 
                " transform_score_by_rating. Use fit_score method to fit the scores"
                " of this classifier."
            )
            raise NotFittedError(message)

        if self.score_by_groups_ and groups is None:
            message = (
                "The score was fitted by group, so the groups parameter must be passed"
                " to this method in order for the score to be predicted."
            )
            raise ValueError(message)

        if not self.score_by_groups_ and groups is not None:
            message = (
                "The score was not fitted by group, so the groups parameter must not be passed"
                " to this method in order for the score to be predicted."
            )
            raise ValueError(message)

        if  groups is not None and any(X.index != groups.index):
            message = (
                "X.index must be equal groups.index."
            )
            raise ValueError(message)

        y_proba = self.predict_proba(X)[:, 1]
        y_proba_df = pd.DataFrame({'proba': y_proba}, index=X.index)
        if groups is not None:
            score = pd.Series([0.0]*len(X), index=X.index).astype('float')
            self.__check_unique_groups(list(groups.unique()))
            for group in groups.unique():
                mask = groups==group
                group_score = (
                    1 -
                    self.proba_quantile_transformers_[group]
                    .transform(y_proba_df).proba
                )*1000
                score = score.mask(mask,group_score)
            if transform_score_by_rating:
                ratings = self.calculate_rating(score,groups)
                min_score_rating = pd.Series([0.0]*len(X), index=X.index).astype('float')
                max_score_rating = pd.Series([0.0]*len(X), index=X.index).astype('float')
                displacement = pd.Series([0.0]*len(X), index=X.index).astype('float')
                new_interval_rating = pd.Series([0.0]*len(X), index=X.index).astype('float')
                for group in list(self.score_limits.keys()):
                    min_score_rating_list = [0 if lim == -np.inf else lim
                                            for lim in self.score_limits[group][:-1]]
                    max_score_rating_list = [1000 if lim == np.inf else lim
                                            for lim in self.score_limits[group][1:]]
                    map_label_min_lim = {label: min_lim
                                            for label,min_lim in zip(self.ratings_labels[group],
                                                                    min_score_rating_list)}
                    map_label_max_lim = {label: max_lim
                                            for label,max_lim in zip(self.ratings_labels[group],
                                                                    max_score_rating_list)}
                    map_label_displacement = {label: idx * 1000 / self.n_ratings[group]
                                                    for idx,label
                                                    in enumerate(self.ratings_labels[group])}
                    mask_group = groups==group
                    min_score_rating = min_score_rating.mask(mask_group,
                                                             ratings.map(map_label_min_lim))
                    max_score_rating = max_score_rating.mask(mask_group,
                                                             ratings.map(map_label_max_lim))
                    displacement = displacement.mask(mask_group,
                                                     ratings.map(map_label_displacement))
                    new_interval_rating = new_interval_rating.mask(mask_group,
                                                                   1000/self.n_ratings[group])

                transformed_score = (
                    (score - min_score_rating) /
                    (max_score_rating-min_score_rating)*new_interval_rating +
                    displacement
                )
                return transformed_score.rename("score")
            else:
                return score.rename("score")
        else:
            score = (
                1 -
                self.proba_quantile_transformers_['unique_group'].transform(y_proba_df).proba            
            )*1000
            if transform_score_by_rating:
                ratings = self.calculate_rating(score,groups)
                min_score_rating_list = [0 if lim == -np.inf else lim
                                        for lim in self.score_limits['unique_group'][:-1]]
                max_score_rating_list = [1000 if lim == np.inf else lim
                                        for lim in self.score_limits['unique_group'][1:]]
                map_label_min_lim = {label: min_lim
                                        for label,min_lim
                                            in zip(self.ratings_labels['unique_group'],
                                                    min_score_rating_list)}
                map_label_max_lim = {label: max_lim
                                        for label,max_lim
                                            in zip(self.ratings_labels['unique_group'],
                                                    max_score_rating_list)}
                map_label_displacement = {label: idx * 1000 / self.n_ratings['unique_group']
                                            for idx,label
                                                in enumerate(self.ratings_labels['unique_group'])}
                min_score_rating = ratings.map(map_label_min_lim).astype(float)
                max_score_rating = ratings.map(map_label_max_lim).astype(float)
                displacement = ratings.map(map_label_displacement).astype(float)
                new_interval_rating = 1000/self.n_ratings['unique_group']

                transformed_score = (
                    (score - min_score_rating) /
                    (max_score_rating-min_score_rating)*new_interval_rating +
                    displacement
                )
                return transformed_score.rename("score")
            else:
                return score.rename("score")

    def calculate_rating(self, score: pd.Series, groups: Union[None, pd.Series]=None):
        """
        Calcula os ratings a partir de um pd.Series de valores de score
        """
        if self.ratings_is_fitted_:
            if not self.score_by_groups_:
                return pd.cut(score, self.score_limits['unique_group'],
                              labels=self.ratings_labels['unique_group'],
                              ordered=False)
            elif groups is not None and self.score_by_groups_:
                self.__check_unique_groups(list(groups.unique()))
                if any(groups.index != score.index):
                    message = (
                        "The index of groups and score must be the same."
                    )
                    raise ValueError(message)
                ratings_series = pd.Series(['E']*len(score), index=score.index)
                for group in list(self.proba_quantile_transformers_.keys()):
                    mask = groups==group
                    faixas_group = pd.cut(score[mask], self.score_limits[group],
                                          labels=self.ratings_labels[group],
                                          ordered=False)
                    ratings_series = ratings_series.mask(mask,faixas_group)
                return ratings_series
            elif self.score_by_groups_ and groups is None:
                message = (
                    "As score was fitted by groups, thus groups parameter must be not None."
                )
                raise ValueError(message) 
        else:
            message = (
                "The ratings must be fitted to use this method. Use fit_ratings method " 
                "to fit the ratings of this classifier."
            )
            raise NotFittedError(message)

    def predict_rating(self, X: pd.DataFrame, groups: Union[None, pd.Series]=None):
        """
        Previsão do rating com cálculo do score
        """

        score = self.predict_score(X, groups)
        return self.calculate_rating(score, groups)

