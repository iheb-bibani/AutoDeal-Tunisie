"""Adaptateurs sklearn stables pour des bibliothèques externes.

CatBoost 1.2.8 ne publie pas encore les nouveaux sklearn tags attendus par
scikit-learn 1.7.x dans certains environnements. Ce wrapper respecte l'API
BaseEstimator/RegressorMixin et garde le modèle picklable hors de __main__.
"""
from sklearn.base import BaseEstimator, RegressorMixin


class SklearnCatBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, iterations=500, depth=7, learning_rate=0.05,
                 loss_function="RMSE", random_seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            verbose=False,
            random_seed=self.random_seed,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
