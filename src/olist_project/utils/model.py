import warnings
from typing import List, Dict, Any
from enum import Enum,auto
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder
from feature_engine.imputation import CategoricalImputer, AddMissingIndicator
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)
from sklearn.exceptions import NotFittedError
from scipy.stats import ks_2samp
import mlflow
import optuna
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from tabulate import tabulate
from olist_project.utils import plot_metrics

project_path = '/home/bruno/Documents/Datarisk/Projetos/ipiranga/modelo-credito-rede/'
conf_path = project_path + '/' + settings.CONF_SOURCE
conf_loader = OmegaConfigLoader(conf_source=conf_path, env="local")
RANDOM_STATE = conf_loader["parameters"]['random_state']

class MetricType(Enum):
    ALL = auto()
    TIME_SPLIT_CV_SCORE = auto()
    TRAIN_TEST_CV_SCORE = auto()
    TEST_CV_SCORE = auto()
    TRAIN_TEST_CV_PREDICT = auto()
    TEST_CV_PREDICT = auto()
    TRAIN_TEST_CV_COHORT = auto()
    TEST_CV_COHORT = auto()
    TEST_CV_PREDICT_BY_GROUP = auto()
    def __str__(self):
        return f'{self.name}'

def _get_shap_importances(model, X):

    _, shap_values = (
        plot_metrics.calculate_shap_values(model, X)
    )
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns,vals)),
                                      columns=['col_names','feature_importance_vals'])

    return feature_importance.sort_values(by=['feature_importance_vals'],
                                          ascending=False,inplace=False)

def _ts_cv_score(model,X, y, datetime_series, n_folds,test_size,
                      format_cohort='%Y%m'):
    cv_args = {"test_size": test_size, "n_splits": n_folds}
    cohort_int = datetime_series.dt.strftime(format_cohort).astype(int)
    X_ts = X.set_index(cohort_int)
    cv_ts = GroupTimeSeriesSplit(**cv_args)
    scores = cross_val_score(model, X_ts, y,
                             groups=X_ts.index,
                             scoring='roc_auc', cv=cv_ts)
    mean = scores.mean()
    std = abs(scores.std())
    return mean, std

def _test_cv_score(model,X, y, n_folds):
    skf = StratifiedKFold(n_folds)
    auc_scores = cross_val_score(model, X, y,
                                 scoring='roc_auc',
                                 cv=skf)
    return auc_scores.mean(), auc_scores.std()

def _test_cv_predict(model,X, y, n_folds):
    skf = StratifiedKFold(n_folds)
    y_proba = cross_val_predict(model, X, y,
                                method="predict_proba", cv=skf)[:,1]
    auc_test = roc_auc_score(y,y_proba)
    return auc_test

def _test_cv_predict_by_group(model,X, y, n_folds, group):
    skf = StratifiedKFold(n_folds)
    y_proba = cross_val_predict(model, X, y,
                                method="predict_proba", cv=skf)[:,1]
    df_res = pd.DataFrame({'target':y,'proba':y_proba, group.name: group})
    auc_test = (
        df_res
        .groupby(group.name)
        .apply(lambda x: pd.Series(
            {'AUC': roc_auc_score(x['target'],x['proba']) if x['target'].nunique() > 1 else 0.0}
        ))
    )
    return auc_test.AUC.mean(), auc_test.AUC.std()

def _y_proba_train(model,X,y):
    try:
        check_is_fitted(model)
    except NotFittedError:
        model.fit(X,y)
    y_proba_train = model.predict_proba(X)[:,1]
    return y_proba_train

def _train_test_cv_predict(model,X, y, n_folds):
    y_proba_train = _y_proba_train(model,X,y)
    auc_train = roc_auc_score(y,y_proba_train)
    auc_test = _test_cv_predict(model,X,y,n_folds)
    mean = (auc_train + auc_test)/2
    std = np.std([auc_train,auc_test])
    return mean, std

def _calculate_roc_auc_train(x):
    return roc_auc_score(x['target'],x['proba_train'])

def _calculate_train_metric_by_cohort(model,X, y, datetime_series):
    y_proba_train = _y_proba_train(model,X,y)
    df_results = pd.DataFrame({'safra': datetime_series,
                               'proba_train': y_proba_train, 
                               'target': y})
    df_results = df_results.groupby('safra').apply(
        lambda x: pd.Series({'auc': _calculate_roc_auc_train(x)})
    )
    return df_results

def _calculate_roc_auc_test(x):
    return roc_auc_score(x['target'],x['proba_test'])
def _calculate_test_metric_by_cohort(model,X, y, datetime_series, n_folds):
    skf = StratifiedKFold(n_folds)
    y_proba_test = cross_val_predict(model, X, y,
                                     method="predict_proba",
                                     cv=skf)[:,1]
    df_results = pd.DataFrame({'safra': datetime_series,
                               'proba_test': y_proba_test,
                               'target': y})
    df_results = df_results.groupby('safra').apply(
        lambda x: pd.Series({'auc': _calculate_roc_auc_test(x)})
    )
    return df_results


def _test_cv_cohort(model,X, y, datetime_series, n_folds):
    df_results_test = _calculate_test_metric_by_cohort(model,X, y, datetime_series, n_folds)
    mean = df_results_test['auc'].mean()
    std = abs(df_results_test['auc'].std())
    return mean, std

def _train_test_cv_cohort(model,X, y, datetime_series, n_folds):
    df_results_train = _calculate_train_metric_by_cohort(model,X, y, datetime_series)
    df_results_test = _calculate_test_metric_by_cohort(model,X, y, datetime_series, n_folds) 
    mean = (df_results_train['auc'] + df_results_test['auc'])/2
    std = abs(np.std([df_results_train['auc'],df_results_test['auc_train']]))
    return mean, std

def _calculate_final_metric(mean,std,with_std_penalization):
    metric = mean - 2*std if with_std_penalization else mean
    if isinstance(mean,pd.Series) and isinstance(std,pd.Series):
        return metric.mean()
    elif isinstance(mean,float) and isinstance(std,float):
        return metric
    else:
        raise TypeError('mean and std types must be pd.Series or float')

class ModelType(Enum):
    """
    """
    LGBM = "lgbm"
    XGB = "xgb"
    CAT = "cat"
    RF = "random_forest"
    DT = "decision_tree"
    LOGREG = "logistic_regression"

    def get_model(self, params):
        if self == ModelType.LGBM:
            return LGBMClassifier(verbosity=-1, random_state=RANDOM_STATE, **params)
        elif self == ModelType.XGB:
            return XGBClassifier(verbosity=0, random_state=RANDOM_STATE, **params)
        elif self == ModelType.CAT:
            return CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, **params)
        elif self == ModelType.RF:
            return RandomForestClassifier(verbose=0, random_state=RANDOM_STATE, **params)
        elif self == ModelType.DT:
            return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
        elif self == ModelType.LOGREG:
            return LogisticRegression(verbose=0, random_state=RANDOM_STATE, **params)

    def get_default_hyperparameter_space(self, trial):
        if self == ModelType.LGBM:
            return {
                'is_unbalance': trial.suggest_categorical('is_unbalance', [True, False]),
                'n_estimators': trial.suggest_int('n_estimators', 2, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.9, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 200),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10000.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1000.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 5, 300),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
                'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 60),
            }
        elif self == ModelType.XGB:
            return {
                'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 250),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 25, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.9, log=True),
            }
        elif self == ModelType.CAT:
            return {
                "iterations": 1000,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "depth": trial.suggest_int("depth", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            }

    def __str__(self):
        return self.value

def _inf_to_nan(X):
    return X.replace([np.inf, -np.inf], np.nan)

def get_model(X_dev,
              cat_vars: List[str]=None,
              num_vars: List[str]=None,
              params: Dict[str, Any]=None,
              model_type: ModelType = ModelType.LGBM,
              encoder = None,
              numerical_imputer = None,):

    if cat_vars is None:
        cat_vars = [col for col in X_dev.select_dtypes('category').columns]

    if num_vars is None:
        num_vars = [col for col in X_dev.select_dtypes('number').columns]

    if set(cat_vars + num_vars) != set(X_dev.columns):
        message = (
            "cat_vars and num_vars must the have the same features as those in X_dev"
        )
        raise ValueError(message)

    if params is None:
        params = dict()

    pipe_steps = []
    if model_type == ModelType.XGB:
        inf_to_nan_transformer = FunctionTransformer(_inf_to_nan)
        pipe_steps.append(
            ('inf_to_nan_transformer',
             SklearnTransformerWrapper(transformer=inf_to_nan_transformer))
        )
    if cat_vars:
        pipe_steps.append(('cat_imputer', CategoricalImputer(variables=cat_vars)))
        pipe_steps.append(('rare_encoder', RareLabelEncoder(variables=cat_vars)))

    if numerical_imputer:
        if model_type == ModelType.LOGREG:
            pipe_steps.append(('missing_indicator', AddMissingIndicator()))
        pipe_steps.append(('num_imputer', numerical_imputer))

    if cat_vars and (encoder is not None):
        pipe_steps.append(
            ('encoder', encoder),
        )
    if cat_vars and model_type == ModelType.CAT:
        params['cat_features'] = cat_vars
    pipe_steps.append(('estimator',model_type.get_model(params)))

    model = Pipeline(pipe_steps)

    return model

def objective(trial, X_dev, y_dev, cohort_dev=None,
              validation_type=MetricType.TEST_CV_PREDICT,
              cv_n_folds=5, ts_val_test_size=3,
              get_model_function = get_model,
              model_type = ModelType.LGBM,
              get_hyperparameters_function=None,
              with_std_penalization=False,
              feature_selection=False,
              min_n_features=10,
              max_shap_sample = 20000,
              performance_group=None
              ):

    if get_hyperparameters_function is not None:
        params = get_hyperparameters_function(trial)
    else:
        params = model_type.get_default_hyperparameter_space(trial)

    if validation_type == MetricType.TEST_CV_PREDICT_BY_GROUP and performance_group is None:
        message = (
            "validation_type is TEST_CV_PREDICT_BY_GROUP but performance_group is None. "
            "You must pass performance_group parameter if validation_type is "
            "TEST_CV_PREDICT_BY_GROUP."
        )
        raise ValueError(message)

    if feature_selection:
        # Seleção de variáveis
        print('Selecting features...')

        model = get_model_function(X_dev, params=params, model_type=model_type)

        model.fit(X_dev,y_dev)
        if isinstance(model,Pipeline) and hasattr(model[:-1],
                                                  'transform'):
            X_dev_transf = model[:-1].transform(X_dev)
        else:
            X_dev_transf = X_dev
        estimator = model[-1] if isinstance(model,Pipeline) else model
        X_dev_transf_sampled = (
            X_dev_transf.sample(max_shap_sample,
                                random_state=RANDOM_STATE)
                if len(X_dev_transf) > max_shap_sample
                    else X_dev_transf
        )
        feature_importance = _get_shap_importances(estimator, X_dev_transf_sampled)
        n_features = trial.suggest_int('n_features',min_n_features,X_dev_transf.shape[1])
        sel_features = list(
            feature_importance
            .sort_values('feature_importance_vals',
                         ascending=False)
            ['col_names'].values[0:n_features]
        )
        trial.set_user_attr("sel_features", sel_features)
    else:
        sel_features = list(X_dev.columns)
    X_dev_new = X_dev[sel_features]

    model = get_model_function(X_dev_new, params=params, model_type=model_type)

    print('Calculating objective metric...')
    if str(validation_type) in ['TIME_SPLIT_CV_SCORE', 'ALL']:
        mean_ts, std_ts = _ts_cv_score(model=model,
                                            X=X_dev_new,
                                            y=y_dev,
                                            datetime_series=cohort_dev,
                                            format_cohort='%Y%m',
                                            n_folds=cv_n_folds,
                                            test_size=ts_val_test_size)
        metric_ts = _calculate_final_metric(mean_ts,
                                            std_ts,
                                            with_std_penalization)
        metric = metric_ts
    if str(validation_type) in ['TEST_CV_COHORT',  'ALL']:
        mean_test_cohort, std_test_cohort = _test_cv_cohort(model=model,
                                                            X=X_dev_new,
                                                            y=y_dev,
                                                            datetime_series=cohort_dev,
                                                            n_folds=cv_n_folds)
        metric_test_cohort = _calculate_final_metric(mean_test_cohort,
                                                     std_test_cohort,
                                                     with_std_penalization)
        metric = metric_test_cohort
    if str(validation_type) in ['TRAIN_TEST_CV_COHORT', 'ALL']:
        mean_train_test_cohort, std_train_test_cohort = _train_test_cv_cohort(
            model=model,
            X=X_dev_new,
            y=y_dev,
            datetime_series=cohort_dev,
            n_folds=cv_n_folds
        )
        metric_train_test_cohort = _calculate_final_metric(mean_train_test_cohort,
                                                           std_train_test_cohort,
                                                           with_std_penalization)
        metric = metric_train_test_cohort

    if str(validation_type) in ['TEST_CV_SCORE', 'ALL']:
        mean_test_cv_score, std_test_cv_score = (
            _test_cv_score(model=model,
                           X=X_dev_new,
                           y=y_dev,
                           n_folds=cv_n_folds)
        )
        metric_test_cv_score = _calculate_final_metric(mean_test_cv_score,
                                                       std_test_cv_score,
                                                       with_std_penalization)
        metric = metric_test_cv_score
    if str(validation_type) in ['TRAIN_TEST_CV_PREDICT', 'ALL']:
        mean_train_test_val, std_train_test_val = (
            _train_test_cv_predict(model=model,
                                              X=X_dev_new,
                                              y=y_dev,
                                              n_folds=cv_n_folds)
        )
        metric_train_test_val = _calculate_final_metric(mean_train_test_val,
                                                        std_train_test_val,
                                                        with_std_penalization)
        metric = metric_train_test_val
    if str(validation_type) in ['TEST_CV_PREDICT', 'ALL']:
        mean_test_val = _test_cv_predict(model=model,
                                                    X=X_dev_new,
                                                    y=y_dev,
                                                    n_folds=cv_n_folds)
        std_test_val = 0.0
        metric_test_val = _calculate_final_metric(mean_test_val,
                                                  std_test_val,
                                                  with_std_penalization)
        metric = metric_test_val
    if str(validation_type) in ['TEST_CV_PREDICT_BY_GROUP', 'ALL']:
        mean_test_val, std_test_val = _test_cv_predict_by_group(model=model,
                                                                X=X_dev_new,
                                                                y=y_dev,
                                                                n_folds=cv_n_folds,
                                                                group=performance_group)
        metric_test_val = _calculate_final_metric(mean_test_val,
                                                  std_test_val,
                                                  with_std_penalization)
        metric = metric_test_val
    if str(validation_type) == 'ALL':
        return (metric_ts, metric_train_test_val,
                metric_test_val, metric_test_cohort,
                metric_train_test_cohort)
    else:
        return metric

def _calculate_results(y_true,y_proba):
    df_results = (
        pd.DataFrame({'target':y_true,'proba':y_proba})
        .sort_values('proba',ascending=True)
        .reset_index(drop=True)
        .assign(
            approval_rate = lambda df: (df.index)/len(df),
            default_rate = lambda df: df.target.cumsum()/(df.index+1)
        )
        .sort_values('proba',ascending=False)
        .reset_index(drop=True)
        .assign(
            cum_true_positive = lambda df: df.target.cumsum(),
            cum_false_positive = lambda df: df.index + 1 - df.cum_true_positive,
            cum_true_negative = lambda df: len(df) - df.target.sum() - df.cum_false_positive,
            cum_false_negative = lambda df: df.target.sum()-df.cum_true_positive,
            accuracy = lambda df: (df.cum_true_positive + df.cum_true_negative) / len(df),
            precision = lambda df: (
                (df.cum_true_positive) / (df.cum_true_positive+df.cum_false_positive)
            ),
            precision_0 = lambda df: (
                (df.cum_true_negative) / (df.cum_true_negative+df.cum_false_negative)
            ),
            recall = lambda df: (
                (df.cum_true_positive) / (df.cum_true_positive+df.cum_false_negative)
            ),
            recall_0 = lambda df: (
                (df.cum_true_negative) / (df.cum_true_negative+df.cum_false_positive)
            ),
            tpr = lambda df: df.recall,
            fpr = lambda df: (
                (df.cum_false_positive) / (df.cum_true_negative+df.cum_false_positive)
            ),
        )
        .sort_values('proba',ascending=True)
        .reset_index(drop=True)
    )
    return df_results

def calculate_uniform_score(y_proba_train, y_proba_test, y_proba_val,random_state):
    df_train = pd.DataFrame({'proba': y_proba_train})
    df_test = pd.DataFrame({'proba': y_proba_test})
    df_val = pd.DataFrame({'proba': y_proba_val})

    quant_transf = QuantileTransformer(random_state=random_state)
    quant_transf.fit(df_train[['proba']])
    df_train['score'] = (1-quant_transf.transform(df_train[['proba']])[:,0])*1000
    df_test['score'] = (1-quant_transf.transform(df_test[['proba']])[:,0])*1000
    df_val['score'] = (1-quant_transf.transform(df_val[['proba']])[:,0])*1000
    return df_train['score'], df_test['score'], df_val['score']

def define_percentile_ratings(y_score_train, y_score_test, y_score_val,
                              n_percentiles):
    df_train = pd.DataFrame({'score': y_score_train})
    df_test = pd.DataFrame({'score': y_score_test})
    df_val = pd.DataFrame({'score': y_score_val})

    disc = EqualFrequencyDiscretiser(q=n_percentiles,
                                     return_boundaries=True,
                                     precision=1)
    disc.fit(df_train[['score']])
    df_train['ratings'] = disc.transform(df_train[['score']])['score']
    df_test['ratings'] = disc.transform(df_test[['score']])['score']
    df_val['ratings'] = disc.transform(df_val[['score']])['score']
    return df_train['ratings'], df_test['ratings'], df_val['ratings']

def _calculate_performance_by_group(target,proba,groups):
    roc_auc_func = lambda x: roc_auc_score(x.target,x.probas)
    ks_func = lambda x: ks_2samp(x.probas[x.target == 0],
                                      x.probas[x.target == 1]).statistic
    df_performance_groups = (
        pd.DataFrame({'target': target,'probas': proba,
                      f'{groups.name}': groups})
        .groupby([f'{groups.name}'])
        .apply(lambda x: pd.Series({'AUC': roc_auc_func(x),
                                    'KS':  ks_func(x)}),
                include_groups=False)
        .reset_index()
        .sort_values(f'{groups.name}')
    )
    return df_performance_groups

def mlflow_experiment_run_cv(model, X_dev, X_oot, y_dev, y_oot,
                                 cohort_dev, cohort_oot, 
                                 features = None, 
                                 n_folds=5, n_percentiles=9,
                                 optuna_study=None,
                                 log_datasets=False,
                                 log_model=False,
                                 log_features=False,
                                 metric_plots=True,
                                 shap_plots=True,
                                 learning_curve_plot=True,
                                 approval_rate_threshold=0.85,
                                 cat_group_dev=None,
                                 cat_group_oot=None,
                                 mlflow_log = False,
                                 run_name=None,
                                 run_id=None,
                                 nested_run = False,
                                 random_state=42,
                                 group_dev=None,
                                 group_oot=None):

    if mlflow_log:
        mlflow.start_run(run_name=run_name,
                         run_id=run_id,
                         nested=nested_run)
        mlflow.set_tag("DS", "Bruno-Souza")

    X_dev_new, X_oot_new = (
        (X_dev[features], X_oot[features])
        if features else (X_dev, X_oot)
    )
    print(f'Number of features: {X_dev_new.shape[1]}')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Calculating cross validation metrics...')
        skf = StratifiedKFold(n_folds)
        y_probas_dev = cross_val_predict(model, X_dev_new, y_dev,
                                         method='predict_proba', cv=skf)[:,1]
        auc_dev = roc_auc_score(y_dev, y_probas_dev)
        aucpr_dev = average_precision_score(y_dev, y_probas_dev)
        ks_dev = ks_2samp(y_probas_dev[y_dev == 0], y_probas_dev[y_dev == 1]).statistic
        brier_loss_dev = brier_score_loss(y_dev, y_probas_dev)
        log_loss_dev = log_loss(y_dev, y_probas_dev)
        print('Calculating training metrics...')
        model.fit(X_dev_new,y_dev)
        y_probas_train = model.predict_proba(X_dev_new)[:,1]
        auc_train = roc_auc_score(y_dev,y_probas_train)
        ks_train = ks_2samp(y_probas_train[y_dev == 0], y_probas_train[y_dev == 1]).statistic
        aucpr_train = average_precision_score(y_dev, y_probas_train)
        brier_loss_train = brier_score_loss(y_dev, y_probas_train)
        log_loss_train = log_loss(y_dev, y_probas_train)

        print('Calculating oot metrics...')
        y_probas_oot = model.predict_proba(X_oot_new)[:,1]
        auc_oot = roc_auc_score(y_oot,y_probas_oot)
        ks_oot = ks_2samp(y_probas_oot[y_oot == 0], y_probas_oot[y_oot == 1]).statistic
        aucpr_oot = average_precision_score(y_oot, y_probas_oot)
        brier_loss_oot = brier_score_loss(y_oot, y_probas_oot)
        log_loss_oot = log_loss(y_oot, y_probas_oot)

    print(f'AUC (dev): {auc_dev}')
    print(f'KS (dev): {ks_dev}')
    print(f'AUCPR (dev): {aucpr_dev}')
    print(f'BRIER LOSS (dev): {brier_loss_dev}')
    print(f'LOG LOSS (dev): {log_loss_dev}')
    print(f'AUC (train): {auc_train}')
    print(f'KS (train): {ks_train}')
    print(f'AUCPR (train): {aucpr_train}')
    print(f'BRIER LOSS (train): {brier_loss_train}')
    print(f'LOG LOSS (train): {log_loss_train}')
    print(f'AUC (oot): {auc_oot}')
    print(f'KS (oot): {ks_oot}')
    print(f'AUCPR (oot): {aucpr_oot}')
    print(f'BRIER LOSS (oot): {brier_loss_oot}')
    print(f'LOG LOSS (oot): {log_loss_oot}')

    if isinstance(cat_group_dev, pd.Series):
        df_performance_tn_dev = (
            _calculate_performance_by_group(y_dev,
                                            y_probas_dev,
                                            cat_group_dev)
        )
        tabular_df = tabulate(df_performance_tn_dev, headers='keys', tablefmt='psql')
        print(tabular_df)
    if isinstance(cat_group_oot, pd.Series):
        df_performance_tn_oot = (
            _calculate_performance_by_group(y_oot,
                                            y_probas_oot,
                                            cat_group_oot)
        )
        tabular_df = tabulate(df_performance_tn_oot, headers='keys', tablefmt='psql')
        print(tabular_df)
    df_results_dev = _calculate_results(y_dev,y_probas_dev)
    results_treshold_dev = (
        df_results_dev
        .query(f'approval_rate>{approval_rate_threshold}')
        .iloc[0]
    )
    acc_dev = results_treshold_dev.accuracy
    recall_dev = results_treshold_dev.recall
    recall_dev_neg = results_treshold_dev.recall_0
    precision_dev = results_treshold_dev.precision
    precision_dev_neg = results_treshold_dev.precision_0
    f1_dev = (2*precision_dev*recall_dev)/(precision_dev+recall_dev)

    df_results_oot = _calculate_results(y_oot,y_probas_oot)
    results_treshold_oot = (
        df_results_oot
        .query(f'approval_rate>{approval_rate_threshold}')
        .iloc[0]
    )
    acc_oot = results_treshold_oot.accuracy
    recall_oot = results_treshold_oot.recall
    recall_oot_neg = results_treshold_oot.recall_0
    precision_oot = results_treshold_oot.precision
    precision_oot_neg = results_treshold_oot.precision_0
    f1_oot = (2*precision_oot*recall_oot)/(precision_oot+recall_oot)
    inad_1 = (
        df_results_oot
        .query(f'approval_rate>{approval_rate_threshold}')
        .target.mean()
    )
    inad_0 = (
        df_results_oot
        .query(f'approval_rate<={approval_rate_threshold}')
        .target.mean()
    )
    print(f'ACC (dev): {acc_dev}')
    print(f'Recall (dev): {recall_dev}')
    print(f'Precision (dev): {precision_dev}')
    print(f'F1 (dev): {f1_dev}')
    print(f'Recall-0 (dev): {recall_dev_neg}')
    print(f'Precision-0 (dev): {precision_dev_neg}')
    print(f'ACC (oot): {acc_oot}')
    print(f'Recall (oot): {recall_oot}')
    print(f'Precision (oot): {precision_oot}')
    print(f'F1 (oot): {f1_oot}')
    print(f'Recall-0 (oot): {recall_oot_neg}')
    print(f'Precision-0 (oot): {precision_oot_neg}')
    print(f'Default Rate - 1 (oot): {inad_1}')
    print(f'Default Rate - 0 (oot): {inad_0}')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if metric_plots:
            print('Generating metrics plots...')
            score_train, score_dev, score_oot = (
                calculate_uniform_score(y_probas_train, y_probas_dev, y_probas_oot,
                                        random_state)
            )
            ratings_train, ratings_dev, ratings_oot = (
                define_percentile_ratings(score_train, score_dev, score_oot,
                                        n_percentiles)
            )
            metrics_figs = (
                plot_metrics.train_test_validation_metrics_new(y_dev, y_dev, y_oot,
                                                               y_probas_train,
                                                               y_probas_dev,
                                                               y_probas_oot,
                                                               cohort_dev,
                                                               cohort_dev,
                                                               cohort_oot,
                                                               ratings_train,
                                                               ratings_dev,
                                                               ratings_oot,
                                                               return_figures=mlflow_log,
                                                               group_train=group_dev,
                                                               group_test=group_dev,
                                                               group_val=group_oot)
            )

        if shap_plots:
            print('Generating shap plots...')
            X_dev_new_sampled = X_dev_new.sample(20000) if len(X_dev_new) > 20000 else X_dev_new
            X_oot_new_sampled = X_oot_new.sample(20000) if len(X_oot_new) > 20000 else X_oot_new
            shap_figs = plot_metrics.train_test_validation_shap_graph(model,
                                                                      X_dev_new_sampled,
                                                                      X_dev_new_sampled,
                                                                      X_oot_new_sampled,
                                                                      return_figure=mlflow_log)

        if learning_curve_plot:
            print('Generating learning curve...')
            learning_curve_fig = plot_metrics.learning_curve_plot(model, X_dev_new, y_dev, skf,
                                                             return_fig=mlflow_log,)

    if mlflow_log:

        if metric_plots:
            for fig_name, fig in metrics_figs.items():
                mlflow.log_figure(fig, f'metrics/{fig_name}')
        if shap_plots:
            for fig_name, fig in shap_figs.items():
                mlflow.log_figure(fig, f'shap/{fig_name}')
        if learning_curve_plot:
            for fig_name, fig in learning_curve_fig.items():
                mlflow.log_figure(fig, f'learning_curve/{fig_name}')

        mlflow.log_metric("auc-dev", auc_dev)
        mlflow.log_metric("ks-dev", ks_dev)
        mlflow.log_metric("aucpr-dev", aucpr_dev)
        mlflow.log_metric("brierloss-dev", brier_loss_dev)
        mlflow.log_metric("logloss-dev", log_loss_dev)
        mlflow.log_metric("auc-train", auc_train)
        mlflow.log_metric("ks-train", ks_train)
        mlflow.log_metric("aucpr-train", aucpr_train)
        mlflow.log_metric("brierloss-train", brier_loss_train)
        mlflow.log_metric("logloss-train", log_loss_train)
        mlflow.log_metric("auc-oot", auc_oot)
        mlflow.log_metric("ks-oot", ks_oot)
        mlflow.log_metric("aucpr-oot", aucpr_oot)
        mlflow.log_metric("brier_loss-oot", brier_loss_oot)
        mlflow.log_metric("log_loss-oot", log_loss_oot)
        mlflow.log_metric("acc-oot", acc_oot)
        mlflow.log_metric("precision_1-oot", precision_oot)
        mlflow.log_metric("precision_0-oot", precision_oot_neg)
        mlflow.log_metric("recall_1-oot", recall_oot)
        mlflow.log_metric("recall_0-oot", recall_oot_neg)
        mlflow.log_metric("inad_1-oot", inad_1)
        mlflow.log_metric("inad_0-oot", inad_0)

        if isinstance(cat_group_dev, pd.Series):
            for cat in df_performance_tn_dev[cat_group_dev.name].unique():
                auc = df_performance_tn_dev.query(f'{cat_group_dev.name}=="{cat}"')['AUC'].iloc[0]
                ks = df_performance_tn_dev.query(f'{cat_group_dev.name}=="{cat}"')['KS'].iloc[0]
                mlflow.log_metric(f"auc-dev_{cat}", auc)
                mlflow.log_metric(f"ks-dev_{cat}", ks)
        if isinstance(cat_group_oot, pd.Series):
            for cat in df_performance_tn_oot[cat_group_oot.name].unique():
                auc = df_performance_tn_oot.query(f'{cat_group_oot.name}=="{cat}"')['AUC'].iloc[0]
                ks = df_performance_tn_oot.query(f'{cat_group_oot.name}=="{cat}"')['KS'].iloc[0]
                mlflow.log_metric(f"auc-oot_{cat}", auc)
                mlflow.log_metric(f"ks-oot_{cat}", ks)

        if log_datasets:
            df_dev = pd.concat([cohort_dev.reset_index(drop=True),
                                X_dev_new.reset_index(drop=True),
                                y_dev.reset_index(drop=True)], axis=1)
            df_oot = pd.concat([cohort_oot.reset_index(drop=True),
                                X_oot_new.reset_index(drop=True),
                                y_oot.reset_index(drop=True)], axis=1)
            data_dev = mlflow.data.pandas_dataset.from_pandas(df_dev, targets=y_dev.name,
                                                              name='dev_dataset')
            data_oot = mlflow.data.pandas_dataset.from_pandas(df_oot, targets=y_oot.name,
                                                              name='oot_dataset')
            mlflow.log_input(data_dev,"training")
            mlflow.log_input(data_oot,"test_oot")
        if log_model:
            signature = mlflow.models.infer_signature(X_dev_new, y_probas_dev)
            mlflow.sklearn.log_model(model,'model',signature=signature)
        if log_features:
            features = list(X_dev_new.columns)
            mlflow.log_dict({'feature_list': features}, 'features.json')
    if optuna_study:
        optuna_figs = {}

        param_names = list(optuna_study.best_params.keys())

        optuna_figs['parallel_coordinate'] = (
            optuna.visualization.plot_parallel_coordinate(optuna_study,params=param_names)
        )
        optuna_figs['parallel_coordinate'].show()
        optuna_figs['history'] = optuna.visualization.plot_optimization_history(optuna_study)
        optuna_figs['history'].show()
        optuna_figs['param_importances'] = (
            optuna.visualization.plot_param_importances(optuna_study)
        )
        optuna_figs['param_importances'].show()

        if mlflow_log:
            for fig_name, fig in optuna_figs.items():
                mlflow.log_figure(fig, f'optimization/{fig_name}.html')
    if mlflow_log:
        mlflow.end_run()
