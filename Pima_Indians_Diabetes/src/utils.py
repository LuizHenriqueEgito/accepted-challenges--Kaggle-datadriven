from sklearn.base import BaseEstimator, TransformerMixin 
import numpy as np


class CorrRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.factor = threshold
        
    def correlation_removal(self, X, y=None):
        X_copy = X.copy()
        corr_matrix = X_copy.corr().abs()
        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(np.bool)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_copy.drop(X_copy.columns[to_drop], axis=1)
        return X_copy
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        return X_copy.apply(self.correlation_removal)
