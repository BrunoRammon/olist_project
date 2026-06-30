from types import GeneratorType
from typing import Union
import logging
from optbinning import BinningProcess
from optbinning import Scorecard
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils._tags import Tags, TargetTags
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, _check_feature_names_in, validate_data
from sklearn.linear_model import LogisticRegression
import scipy.stats as stat
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from feature_engine.selection.base_selector import BaseSelector
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from sklearn.model_selection import cross_validate
from sklearn.inspection import permutation_importance
from scipy.stats import norm
from feature_engine.variable_handling import (
    check_numerical_variables,
    retain_variables_if_in_df,
)
from feature_engine.tags import _return_tags


class CustomBinaryScorecard(Scorecard):
    def __init__(self, binning_process, estimator, scaling_method='min_max',
                 scaling_method_params={'min': 0, 'max': 1000}, intercept_based=False,
                 reverse_scorecard=False, rounding=False, verbose=False):
        super().__init__(
            binning_process, estimator, scaling_method=scaling_method,
            scaling_method_params=scaling_method_params, intercept_based=intercept_based,
            reverse_scorecard=reverse_scorecard, rounding=rounding, verbose=verbose
        )

    def fit(self, X, y, sample_weight=None, metric_special=0, metric_missing=0,
            show_digits=4, check_input=False):
        self.binning_process.variable_names = list(X.columns)
        self.feature_names_in_ = list(X.columns)
        return_obj = super().fit(X, y, sample_weight, metric_special, metric_missing,
                                 show_digits, check_input)
        self.classes_ = np.unique(y)
        self._is_fitted = True
        self.coef_ = self.estimator_.coef_
        return return_obj

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
    
    def __sklearn_tags__(self):
        return Tags(
            estimator_type='classifier',
            target_tags=TargetTags(required=False),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
        )
    
    @property
    def feature_importances_(self):
        assert self._is_fitted
        feature_importances = (
            self.table('summary')
            .groupby('Variable', as_index=False, sort=False)
            .agg(
                point_min = ('Points', 'min'),
                point_max = ('Points', 'max')
            )
            .assign(
                feature_importance = lambda df: df.point_max - df.point_min
            )
            .rename(columns={'Variable': 'feature'})
            [['feature', 'feature_importance']]
            .feature_importance
            .tolist()
        )
        return feature_importances


class CustomBinningProcess(BinningProcess):
    def __init__(self, variable_names=[''], max_n_prebins=20, min_prebin_size=0.05,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, max_pvalue=None,
                 max_pvalue_policy="consecutive", selection_criteria=None,
                 fixed_variables=None, categorical_variables=None,
                 special_codes=None, split_digits=None,
                 binning_fit_params=None, binning_transform_params=None,
                 n_jobs=None, verbose=False):

        super().__init__(variable_names, max_n_prebins, 
                         min_prebin_size, 
                         min_n_bins, 
                         max_n_bins, 
                         min_bin_size, 
                         max_bin_size, 
                         max_pvalue, 
                         max_pvalue_policy, 
                         selection_criteria, 
                         fixed_variables, 
                         categorical_variables, 
                         special_codes, 
                         split_digits, 
                         binning_fit_params, 
                         binning_transform_params, 
                         n_jobs, 
                         verbose)

    def transform(self, X, metric=None, metric_special=0, metric_missing='empirical',
                show_digits=2, check_input=False):
        """Transform given data to metric using bins from each fitted optimal
        binning.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        X_new : numpy array or pandas.DataFrame, shape = (n_samples,
        n_features_new)
            Transformed array.
        """
        return super().transform(X, metric, metric_special, metric_missing,
                                 show_digits, check_input)
    
    def fit(self, X, y, sample_weight=None, check_input=False):
        """Fit the binning process. Fit the optimal binning to all variables
        according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

            .. versionchanged:: 0.4.0
            X supports ``numpy.ndarray`` and ``pandas.DataFrame``.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            Only applied if ``prebinning_method="cart"``. This option is only
            available for a binary target.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        self.variable_names = list(X.columns)
        fitted_obj = super()._fit(X, y, sample_weight, check_input)
        return fitted_obj

    def transform(self, X, metric=None, metric_special=0, metric_missing='empirical',
                show_digits=2, check_input=False):
        """Transform given data to metric using bins from each fitted optimal
        binning.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        X_new : numpy array or pandas.DataFrame, shape = (n_samples,
        n_features_new)
            Transformed array.
        """
        return super().transform(X, metric, metric_special, metric_missing,
                                 show_digits, check_input)

    def __sklearn_is_fitted__(self):
            """
            Check fitted status and return a Boolean value.
            """
            return hasattr(self, "_is_fitted") and self._is_fitted

class IVFeatureSelection(BinningProcess):
    def __init__(self, variable_names=[''], max_n_prebins=20, min_prebin_size=0.05,
                 min_n_bins=None, max_n_bins=None, min_bin_size=None,
                 max_bin_size=None, max_pvalue=None,
                 max_pvalue_policy="consecutive", selection_criteria=None,
                 fixed_variables=None, categorical_variables=None,
                 special_codes=None, split_digits=None,
                 binning_fit_params=None, binning_transform_params=None,
                 n_jobs=None, verbose=False,
                 iv_min=0.2, iv_max=0.5):

        super().__init__(variable_names, max_n_prebins, 
                         min_prebin_size, 
                         min_n_bins, 
                         max_n_bins, 
                         min_bin_size, 
                         max_bin_size, 
                         max_pvalue, 
                         max_pvalue_policy, 
                         selection_criteria, 
                         fixed_variables, 
                         categorical_variables, 
                         special_codes, 
                         split_digits, 
                         binning_fit_params, 
                         binning_transform_params, 
                         n_jobs, 
                         verbose)
        self._features_to_drop = None
        self._feature_names_out = None
        self.iv_min = iv_min
        self.iv_max = iv_max

    def transform(self, X, metric=None, metric_special=0, metric_missing='empirical',
                show_digits=2, check_input=False):
        """Transform given data to metric using bins from each fitted optimal
        binning.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

        metric : str or None, (default=None)
            The metric used to transform the input vector. If None, the default
            transformation metric for each target type is applied. For binary
            target options are: "woe" (default), "event_rate", "indices" and
            "bins". For continuous target options are: "mean" (default),
            "indices" and "bins". For multiclass target options are:
            "mean_woe" (default), "weighted_mean_woe", "indices" and "bins".

        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate for a binary target, and any numerical value for other
            targets.

        show_digits : int, optional (default=2)
            The number of significant digits of the bin column. Applies when
            ``metric="bins"``.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        X_new : numpy array or pandas.DataFrame, shape = (n_samples,
        n_features_new)
            Transformed array.
        """
        return super().transform(X, metric, metric_special, metric_missing,
                                 show_digits, check_input)
    
    def fit(self, X, y, sample_weight=None, check_input=False):
        """Fit the binning process. Fit the optimal binning to all variables
        according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples.

            .. versionchanged:: 0.4.0
            X supports ``numpy.ndarray`` and ``pandas.DataFrame``.

        y : array-like of shape (n_samples,)
            Target vector relative to x.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            Only applied if ``prebinning_method="cart"``. This option is only
            available for a binary target.

        check_input : bool (default=False)
            Whether to check input arrays.

        Returns
        -------
        self : BinningProcess
            Fitted binning process.
        """
        self.variable_names = list(X.columns)
        fitted_obj = super()._fit(X, y, sample_weight, check_input)

        iv_selection = (
            fitted_obj.summary()
            .query(f'iv.between({self.iv_min},{self.iv_max})').name.unique()
        )
        self._features_to_drop = [feat for feat in X.columns if feat not in iv_selection]
        self._feature_names_out = [feat for feat in X.columns if feat in iv_selection]
        return fitted_obj

    def get_feature_names_out(self):
        return self._feature_names_out
    
    def get_features_to_drop(self):
        return self._features_to_drop

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            if key not in ['iv_min', 'iv_max']:
                out[key] = value
        return out

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


class LogisticInferenceMixin:
    """
    Adds standard errors, z-scores and p-values to a fitted sklearn LogisticRegression.
    """

    def _robust_covariance(self, X, y):
        decision = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-decision))
        residuals = np.asarray(y - p)

        score = X * residuals[:, None]
        meat = score.T @ score

        F = self._compute_fisher_information(X)
        bread = np.linalg.pinv(F)

        return bread @ meat @ bread

    def _compute_fisher_information(self, X):
        decision = np.asarray(self.decision_function(X))
        p = 1.0 / (1.0 + np.exp(-decision))
        W = p * (1 - p)

        # Fisher Information: X^T W X
        X_weighted = X * W[:, None]
        return X_weighted.T @ X

    def compute_inference(self, X, y, robust=False):

        if robust:
            cov = self._robust_covariance(X, y)
        else:
            F = self._compute_fisher_information(X)
            cov = np.linalg.pinv(F)

        self.covariance_ = cov
        self.std_errors_ = np.sqrt(np.diag(cov))
        self.z_scores_ = self.coef_[0] / self.std_errors_
        self.p_values_ = 2 * stats.norm.sf(np.abs(self.z_scores_))

        return self

class CustomLogisticRegression(LogisticRegression, LogisticInferenceMixin):
    def __init__(
        self,
        *,
        dual=False,
        tol=1e-4,
        fit_intercept=False,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        verbose=0,
        warm_start=False,
        n_jobs=None
    ):
        super().__init__(
            C=np.inf,
            l1_ratio=0.0,
            dual=dual,
            tol=tol,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )

    def fit(self,X,y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self.compute_inference(X, y, robust=True)
        return self

class SMLogitClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper for statsmodels Logit with calibration support
    """

    def __init__(
        self,
        method="newton",
        maxiter=1000,
        add_constant=False,
        cov_type="nonrobust",
        cov_kwds={},
    ):
        self.method = method
        self.maxiter = maxiter
        self.add_constant = add_constant
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_X(self, X):
        X = check_array(X)
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        if self.add_constant:
            X = sm.add_constant(X, has_constant="add")
        return X

    def _validate_features(self, X):
        if hasattr(X, "columns"):
            input_features = np.asarray(X.columns)
        else:
            input_features = None
        _check_feature_names_in(
            self, input_features
        )
    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        # --- feature names (sklearn convention) ---
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None

        X, y = check_X_y(X, y)
        X = pd.DataFrame(X, columns=self.feature_names_in_)

        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("SMLogitClassifier supports binary classification only.")

        X_sm = self._prepare_X(X)

        self.model_ = sm.Logit(y, X_sm)
        kwargs = {} if self.cov_kwds else self.cov_kwds
        self.result_ = self.model_.fit(
            disp=False,
            method=self.method,
            maxiter=self.maxiter,
            cov_type=self.cov_type,
            **kwargs
        )

        params = self.result_.params.values

        if self.add_constant:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:].reshape(1, -1)
        else:
            self.intercept_ = 0.0
            self.coef_ = params.reshape(1, -1)

        return self

    def decision_function(self, X):
        check_is_fitted(self, "result_")
        self._validate_features(X)

        X_sm = self._prepare_X(X)
        return X_sm.dot(self.result_.params).to_numpy()

    def predict_proba(self, X):
        check_is_fitted(self, "result_")
        self._validate_features(X)

        X_sm = self._prepare_X(X)
        probs = self.result_.predict(X_sm)

        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return self.classes_[(probs >= 0.5).astype(int)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # ------------------------------------------------------------------
    # sklearn metadata
    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "result_")

        if self.feature_names_in_ is None:
            raise ValueError("Estimator was fitted without feature names.")

        return self.feature_names_in_

    # ------------------------------------------------------------------
    # statsmodels diagnostics
    # ------------------------------------------------------------------
    def summary(self):
        check_is_fitted(self, "result_")
        return self.result_.summary()

    @property
    def pvalues_(self):
        check_is_fitted(self, "result_")
        return self.result_.pvalues

    @property
    def conf_int_(self):
        check_is_fitted(self, "result_")
        return self.result_.conf_int()

    @property
    def standard_errors_(self):
        check_is_fitted(self, "result_")
        return self.result_.bse

import logging
from sklearn.model_selection import cross_validate

from feature_engine._docstrings.fit_attributes import (
    _feature_importances_docstring,
    _feature_importances_std_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _performance_drifts_docstring,
    _performance_drifts_std_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _features_to_drop_docstring,
    _fit_docstring,
    _get_support_docstring,
    _groups_docstring,
    _initial_model_performance_docstring,
    _scoring_docstring,
    _threshold_docstring,
    _transform_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.selection.base_recursive_selector import BaseRecursiveSelector
from typing import List, Union


Variables = Union[None, int, str, List[Union[str, int]]]
logger = logging.getLogger(__name__)

@Substitution(
    scoring=_scoring_docstring,
    threshold=_threshold_docstring,
    cv=_cv_docstring,
    groups=_groups_docstring,
    variables=_variables_numerical_docstring,
    confirm_variables=_confirm_variables_docstring,
    initial_model_performance_=_initial_model_performance_docstring,
    feature_importances_=_feature_importances_docstring,
    feature_importances_std_=_feature_importances_std_docstring,
    performance_drifts_=_performance_drifts_docstring,
    performance_drifts_std_=_performance_drifts_std_docstring,
    features_to_drop_=_features_to_drop_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_docstring,
    transform=_transform_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class CustomRecursiveFeatureElimination(BaseRecursiveSelector):
    """
    RecursiveFeatureElimination() selects features following a recursive elimination
    process.

    The process is as follows:

    1. Train an estimator using all the features.

    2. Rank the features according to their importance derived from the estimator.

    3. Remove the least important feature and fit a new estimator.

    4. Calculate the performance of the new estimator.

    5. Calculate the performance difference between the new and original estimator.

    6. If the performance drop is below the threshold the feature is removed.

    7. Repeat steps 3-6 until all features have been evaluated.

    Model training and performance evaluation are done with cross-validation.

    More details in the :ref:`User Guide <recursive_elimination>`.

    Parameters
    ----------
    estimator: object
        A Scikit-learn estimator for regression or classification.

    {variables}

    {scoring}

    {threshold}

    {cv}

    {groups}

    {confirm_variables}

    Attributes
    ----------
    {initial_model_performance_}

    {feature_importances_}

    {feature_importances_std_}

    {performance_drifts_}

    {performance_drifts_std_}

    {features_to_drop_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {get_support}

    {transform}

    Examples
    --------

    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from feature_engine.selection import RecursiveFeatureElimination
    >>> X = pd.DataFrame(dict(x1 = [1000,2000,1000,1000,2000,3000],
    >>>                     x2 = [2,4,3,1,2,2],
    >>>                     x3 = [1,1,1,0,0,0],
    >>>                     x4 = [1,2,1,1,0,1],
    >>>                     x5 = [1,1,1,1,1,1]))
    >>> y = pd.Series([1,0,0,1,1,0])
    >>> rfe = RecursiveFeatureElimination(RandomForestClassifier(random_state=2), cv=2)
    >>> rfe.fit_transform(X, y)
       x2
    0   2
    1   4
    2   3
    3   1
    4   2
    5   2
    """

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        threshold: Union[int, float] = 0.01,
        variables: Variables = None,
        confirm_variables: bool = False,
        verbose: bool = False
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv,
            groups=groups,
            threshold=threshold,
            variables=variables,
            confirm_variables=confirm_variables
        )
        self.verbose = verbose

    def _log(self,message):
        if self.verbose:
            logger.info(message)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe
        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        message = f'Calculating initial feature importances and baseline performance...'
        self._log(message)

        X, y = super().fit(X, y)

        # Sort the feature importance values increasingly
        self.feature_importances_.sort_values(ascending=True, inplace=True)

        # to collect selected features
        _selected_features = []

        # temporary copy where we will remove features recursively
        X_tmp = X[self.variables_].copy()

        # we need to update the performance as we remove features
        baseline_model_performance = self.initial_model_performance_

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}
        self.performance_drifts_std_ = {}

        # evaluate every feature, starting from the least important
        # remember that feature_importances_ is ordered already
        variable_list = list(self.feature_importances_.index)
        n_vars = len(variable_list)
        for i_var, feature in enumerate(variable_list, start=1):
            
            message = f'Analysing {feature} ({i_var}/{n_vars})...'
            self._log(message)
                
            # if there is only 1 feature left
            if X_tmp.shape[1] == 1:
                self.performance_drifts_[feature] = 0
                _selected_features.append(feature)
                break

            # remove feature and train new model
            model_tmp = cross_validate(
                estimator=self.estimator,
                X=X_tmp.drop(columns=feature),
                y=y,
                cv=self._cv,
                groups=self.groups,
                scoring=self.scoring,
                return_estimator=False,
            )

            # assign new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = baseline_model_performance - model_tmp_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift
            self.performance_drifts_std_[feature] = model_tmp["test_score"].std()

            if performance_drift > self.threshold:
                message = f'Feature {feature} kept: {performance_drift} > {self.threshold}'
                self._log(message)
                _selected_features.append(feature)

            else:
                # remove feature and adjust initial performance
                message = f'Feature {feature} removed: {performance_drift} <= {self.threshold}'
                self._log(message)
                X_tmp = X_tmp.drop(columns=feature)

                # message = f'Adjusting initial performance...'
                # self._log(message)
                # baseline_model = cross_validate(
                #     estimator=self.estimator,
                #     X=X_tmp,
                #     y=y,
                #     cv=self._cv,
                #     groups=self.groups,
                #     scoring=self.scoring,
                #     return_estimator=False,
                # )

                # store initial model performance
                # baseline_model_performance = baseline_model["test_score"].mean()
                baseline_model_performance = model_tmp_performance

        self.features_to_drop_ = [
            f for f in self.variables_ if f not in _selected_features
        ]

        return self


logger = logging.getLogger(__name__)

class CustomBaseRecursiveSelector(BaseSelector):
    """
    Shared functionality for recursive selectors.

    Parameters
    ----------
    estimator: object
        A Scikit-learn estimator for regression or classification.

    variables: str or list, default=None
        The list of variable to be evaluated. If None, the transformer will evaluate
        all numerical features in the dataset.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold can be a big number.
        The threshold must be defined by the user. Bigger thresholds will select less
        features.

    cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use cross_validate's default 5-fold cross validation

            - int, to specify the number of folds in a (Stratified)KFold,

            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)

            - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation.

    groups: Array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting
        the dataset into train/test set. Only used in conjunction with a
        “Group” cv instance (e.g., GroupKFold).

    confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will be
        removed from the list of variables. Only used when passing a variable list to
        the parameter `variables`. See parameter variables for more details.

    Attributes
    ----------
    initial_model_performance_:
        Performance of the model trained using the original dataset.

    feature_importances_:
        Pandas Series with the feature importance (comes from step 2)

    feature_importances_std_:
        Pandas Series with the standard deviation of the feature importance.

    features_to_drop_:
        List with the features to remove from the dataset.

    variables_:
        The variables that will be considered for the feature selection.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find the important features.
    """

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        threshold: Union[int, float] = 0.01,
        variables: Variables = None,
        confirm_variables: bool = False,
    ):

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float")

        super().__init__(confirm_variables)
        self.variables = _check_variables_input_value(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv
        self.groups = groups

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find initial model performance. Sort features by importance.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe

        y: array-like of shape (n_samples)
           Target variable. Required to train the estimator.
        """

        # check input dataframe
        # X, y = check_X_y(X, y)

        if self.variables is None:
            self.variables_ = list(X.columns)
        else:
            if self.confirm_variables is True:
                variables_ = retain_variables_if_in_df(X, self.variables)
                self.variables_ = variables_
            else:
                self.variables_ = self.variables
        self.variables_out_study_ = [var for var in X.columns if var not in self.variables_]

        self._cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        # save input features
        self._get_feature_names_in(X)

        # train model with all features and cross-validation
        return X, y

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

class SmartRFE(CustomBaseRecursiveSelector):
    """
    Recursive Feature Elimination with statistical validation (SmartRFE).

    It removes the least important feature, checks model performance via CV,
    and only permanently drops the feature if the performance loss is
    statistically insignificant and within global bounds.
    """

    def __init__(
        self,
        estimator,
        scoring: str = "roc_auc",
        cv=3,
        groups=None,
        threshold: Union[int, float] = 0.01,
        alpha: float = 0.05,
        max_global_drop: float = 0.02,
        min_features: int = 1,
        variables=None,
        confirm_variables: bool = False,
        verbose: bool = False,
        n_jobs: int = None
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv,
            groups=groups,
            threshold=threshold,
            variables=variables,
            confirm_variables=confirm_variables,
        )
        self.alpha = alpha
        self.max_global_drop = max_global_drop
        self.min_features = min_features
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _log(self, msg):
        if self.verbose:
            logger.info(msg)

    def _get_importances(self, estimator, X, y):
        """
        Extracts feature importances from a fitted estimator.
        Fallback to permutation importance if native attributes are missing.
        """
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_

        if hasattr(estimator, "coef_"):
            imps = np.abs(estimator.coef_)
            if imps.ndim > 1:
                imps = imps[0]
            return imps

        # Fallback: Permutation importance (computationally expensive)
        r = permutation_importance(
            estimator, X, y, n_repeats=1, random_state=10
        )
        return r.importances_mean

    def _get_model_results(self, X, y):
        """
        Fits the model using CV to get raw scores and average feature importances.
        """
        res = cross_validate(
            estimator=self.estimator,
            X=X,
            y=y,
            cv=self._cv,
            groups=self.groups,
            scoring=self.scoring,
            return_estimator=True,
            n_jobs=self.n_jobs
        )
        
        scores = res["test_score"]
        
        # Aggregate importances across folds
        imps_list = []
        for est in res["estimator"]:
            imps_list.append(self._get_importances(est, X, y))
        
        avg_importances = np.mean(imps_list, axis=0)
        importances = pd.Series(avg_importances, index=X.columns)
        
        return scores, importances

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 1. Base Class Init: Checks X/y, defines self.variables_, self._cv
        # Note: This runs an initial CV fit (BaseRecursiveSelector behavior),
        # but we need to re-run it below to capture the *raw scores* for our
        # paired t-tests (Base only stores the mean).
        message = f'Calculating initial feature importances and baseline performance...'
        self._log(message)
        X, y = super().fit(X, y)

        X_current = X[self.variables_ + self.variables_out_study_].copy()
        message = f'Study variables: {self.variables_}'
        self._log(message)
        message = f'Out of study variables: {self.variables_out_study_}'
        self._log(message)
        # 2. Establish Baseline (Raw Scores)
        
        # We re-calculate because we need the raw scores array, not just the mean
        fixed_scores, current_importances = self._get_model_results(X_current, y)
        
        # Set baselines
        rolling_scores = fixed_scores
        # Overwrite initial performance with our fresh calculation to ensure consistency
        self.initial_model_performance_ = fixed_scores.mean()

        # Z-score for one-tailed test (significance level)
        z = norm.ppf(1 - self.alpha)

        protected = set()
        self.elimination_path_ = []
        
        # 3. Iterative Elimination Loop
        while X_current.drop(columns=self.variables_out_study_, errors='ignore').shape[1] > self.min_features:
            
            # Sort features by importance (ascending)
            # features with lowest importance are candidates for removal
            sorted_importances = current_importances.sort_values()
            
            # Find next candidate that isn't protected
            candidates = [f for f in sorted_importances.index if (f not in protected) and (f not in self.variables_out_study_)]

            if not candidates:
                self._log("All remaining features are protected. Stopping.")
                break

            n_candidates = len(candidates)
            feature_to_drop = candidates[0]
            self._log(f"There are {n_candidates} remaining candidates. Testing removal of '{feature_to_drop}'...")
            
            # Create temporary subset
            X_candidate = X_current.drop(columns=[feature_to_drop])
            
            # Evaluate model without the feature
            cand_scores, cand_importances = self._get_model_results(X_candidate, y)
            
            # ---- Statistical Analysis (Paired Differences) ----
            
            # 1. Rolling Drift: Compare against previous step
            # Positive diff => Performance Dropped (Previous > Current)
            diff_rolling = rolling_scores - cand_scores
            mean_drift_rolling = diff_rolling.mean()
            std_error_rolling = diff_rolling.std(ddof=1) / np.sqrt(len(diff_rolling))
            
            # 2. Fixed Drift: Compare against original baseline
            diff_fixed = fixed_scores - cand_scores
            mean_drift_fixed = diff_fixed.mean()
            
            # Calculate allowed drop limit (Threshold + Statistical Margin)
            # If threshold is 0.0, we only allow drops that are statistically insignificant
            limit = self.threshold + (z * std_error_rolling)
            
            is_statistically_safe = mean_drift_rolling <= limit
            is_globally_safe = mean_drift_fixed <= self.max_global_drop
            removed = is_statistically_safe and is_globally_safe
            # Log metrics for inspection
            self.elimination_path_.append({
                "feature": feature_to_drop,
                "drift_rolling": mean_drift_rolling,
                "max_statistical_drop": limit,
                "drift_fixed": mean_drift_fixed,
                "max_global_drop": self.max_global_drop,
                "se_rolling": std_error_rolling,
                "n_features": X_candidate.shape[1],
                "removed": removed
            })
            
            if removed:
                self._log(f"  >> REMOVED '{feature_to_drop}'. Rolling Drift: {mean_drift_rolling:.4f}")
                
                # Commit the change
                X_current = X_candidate
                rolling_scores = cand_scores
                
                # Re-calculate importances for the new subset
                # (The importances shift after a variable is removed)
                current_importances = cand_importances
                
            else:
                reason = "Statistically significant drop" if not is_statistically_safe else "Max global drop exceeded"
                self._log(f"  >> PROTECTED '{feature_to_drop}'. Reason: {reason}")
                protected.add(feature_to_drop)
                # We do NOT update X_current or re-calculate importances here.
                # We simply loop back and pick the next candidate from the *existing* list.

        # 4. Final Cleanup
        self.variables_ = list(X_current.columns)
        self.features_to_drop_ = [f for f in self.feature_names_in_ if f not in self.variables_]
        
        # Update final importances
        self.feature_importances_ = current_importances
        
        return self
