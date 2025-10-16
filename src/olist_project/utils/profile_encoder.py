import pandas as pd
import numpy as np
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.base import _fit_context

class ProfileEncoder(_BaseEncoder):

    _parameter_constraints: dict = {
        "variables": [list],
        "n_groups": [int, None]
    }

    def __init__(self,
                 variables,
                 n_groups=None):
        self.variables=variables
        self.n_groups=n_groups
        self.is_fitted_ = False

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self,X,y):
        if self.n_groups is None:
            assign_dict = {'profile_default_groups': lambda df: df.default}
        else: 
            assign_dict = {
                'default_gt_0': lambda df: df.default.where(df.default>0,np.nan),
                'default_groups_gt_0': lambda df: (
                    pd.qcut(df.default_gt_0, q=self.n_groups-1,
                            labels=False,duplicates='drop').astype(float)
                ),
                'profile_default_groups': lambda df: (
                    df.default.where(df.default==0,
                                    df.default_groups_gt_0+1)

                )
            }
            
        self._default_groups_mapping = (
            X
            .assign(
                target = y
            )
            .groupby(self.variables,observed=True)
            .agg(
                default = ('target','mean'),
            )
            .reset_index()
            .assign(**assign_dict)
            [self.variables+['profile_default_groups']]
            .to_dict('list')
        )
        self.is_fitted_ = True
        return self

    def transform(self,X):
        check_is_fitted(self)
        df_map = pd.DataFrame(self._default_groups_mapping)
        map_dtypes = {var: 'category' for var in self.variables}
        X_transformed = X.merge(df_map, on=self.variables, how='left').astype(map_dtypes)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
