import numpy as np
import pandas as pd
from pandas import Series
from sklearn.base import BaseEstimator,TransformerMixin

def indicies_of_outliers(x: Series):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))[0]

def outliers_z_score(ys: Series):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)[0]

def outliers_modified_z_score(ys: Series):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    
    if median_absolute_deviation_y == 0:
        median_absolute_deviation_y = np.finfo(np.double).min
    
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)[0]

class IQROutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor:float = 1.5):
        self.factor = factor or 1.5
        return

    def __outlierDetector(self, X, y=None):
        X = pd.Series(X).copy()
        q1, q3 = np.percentile(X, [25, 75])
        iqr = q3 - q1
        self.lower_bound.append(q1 - (iqr * self.factor))
        self.upper_bound.append(q3 + (iqr * self.factor))

    def fit(self, X, y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.__outlierDetector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        return X

class ZScoreOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold:float = 3):
        self.threshold = threshold or 3
        return

    def __outlierDetector(self, X, y=None):
        X = pd.Series(X).copy()
        mean_y = np.mean(X)
        stdev_y = np.std(X)
        self.zscores.append([(x - mean_y) / stdev_y for x in X])

    def fit(self, X, y=None):
        self.zscores = []
        X.apply(self.__outlierDetector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            cond = np.abs(self.zscores[i]) > self.threshold
            x[cond] = np.nan
            X.iloc[:, i] = x
        return X

class ModZScoreOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold:float = 3.5):
        self.threshold = threshold or 3.5
        return
        
    def __outlierDetector(self, X, y=None):
        X = pd.Series(X).copy()
        median_y = np.median(X)
        median_absolute_deviation_y = np.median([np.abs(x - median_y) for x in X])
    
        if median_absolute_deviation_y == 0:
            median_absolute_deviation_y = np.finfo(np.double).min

        modified_z_scores = [0.6745 * (x - median_y) / median_absolute_deviation_y for x in X]
        self.zscores.append(modified_z_scores)

    def fit(self, X, y=None):
        self.zscores = []
        X.apply(self.__outlierDetector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[np.abs(self.zscores[i]) > self.threshold] = np.nan
            X.iloc[:, i] = x
        return X

class OutlierRemover(BaseEstimator, TransformerMixin):
    VALID_TYPES = ['IQR','ZScore', 'ModZScore']

    def __generateRemover(type: str, kwargs=None):
        if kwargs is not None:
            match type:
                case 'IQR':
                    return IQROutlierRemover(**kwargs)
                case 'ZScore':
                    return ZScoreOutlierRemover(**kwargs)
                case 'ModZScore':
                    return ModZScoreOutlierRemover(**kwargs)
                case _:
                    return None
        else:
            match type:
                case 'IQR':
                    return IQROutlierRemover()
                case 'ZScore':
                    return ZScoreOutlierRemover()
                case 'ModZScore':
                    return ModZScoreOutlierRemover()
                case _:
                    return None

    def __init__(self, type='IQR', **kwargs):
        if type not in self.VALID_TYPES:
            raise ValueError("Results: status must be one of %r." % self.VALID_TYPES)
        self.type = type
        self.remover = OutlierRemover.__generateRemover(self.type, kwargs)

    def fit(self, X, y=None):
        return self.remover.fit(X,y)

    def transform(self, X, y=None):
        return self.remover.transform(X,y)