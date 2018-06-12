import numpy as np
import pandas as pd
import scipy as sp
import datetime
from scipy import stats

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

def sum_1(x):
    return x + 1

def apply_antilog(x):
    #return np.exp(x) - 1
    return np.exp(x)
    
def apply_log(x):
    #return np.log(x + 1)
    return np.log(x)
    
def all_but_first_column(X):
    return X[:, 1:]

def all_columns(X):
    return X

def apply_power_cube(x):
    return np.power(x, 3)

def apply_cube_root(x):
    return sp.special.cbrt(x)

def apply_reciprocal(x):
    return 1 / x

#def extract_issued_at2(x):
#    return datetime.datetime.strptime(x['issued_at'][0:19],  "%Y-%m-%dT%H:%M:%S")

def extract_issued_at(x):
    return datetime.datetime.strptime(x['data']['issued_at'][0:19],  "%Y-%m-%dT%H:%M:%S")

#def extract_issued_at_str(x):
#    return x['data']['issued_at']

#def extract_issued_at_datetime(x):
#    return datetime.datetime.strptime(x['issued_at_str'][0:19], "%Y-%m-%dT%H:%M:%S")

def bin_data(x, **kw_args):
    columns = kw_args['columns']
    bins = kw_args['bins']
    for index in range(0, len(columns)):
        column = columns[index]
        x[:, index] = np.digitize(x[:, index], bins[column])
    return pd.DataFrame(x).astype(int)

class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))

class DataFrameColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
        return X[self.columns]

    def fit(self, X, y=None, **fit_params):
        return self
    
class ToDictTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        return X.astype(str).to_dict(orient='records')

    def fit(self, X, y=None, **fit_params):
        return self