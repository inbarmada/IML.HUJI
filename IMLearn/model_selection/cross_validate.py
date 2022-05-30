from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_partition = np.array_split(X, cv)
    y_partition = np.array_split(y, cv)

    val_scores = np.array([])
    train_scores = np.array([])
    for i in range(cv):
        X_i = np.array([])
        y_i = np.array([])
        for j in range(cv):
            if j != i:
                X_i = np.hstack([X_i, X_partition[j]])
                y_i = np.hstack([y_i, y_partition[j]])
        estimator.fit(X_i, y_i)

        pred_i = estimator.predict(X_partition[i])
        val_scores = np.hstack([val_scores, scoring(pred_i, y_partition[i])])

        pred_train_i = estimator.predict(X_i)
        train_scores = np.hstack([train_scores, scoring(pred_train_i, y_i)])
    return train_scores.mean(), val_scores.mean()
