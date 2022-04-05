from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from ...metrics import loss_functions
import plotly.graph_objects as go

class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            X = np.c_[X, np.ones(len(X))]

        X_pinv = np.linalg.pinv(X)
        self.coefs_ = np.matmul(X_pinv, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # Add column of ones to X if intercept
        if self.include_intercept_:
            X = np.c_[X, np.ones(len(X))]

        y = np.matmul(X, self.coefs_)
        return y

    def _loss(self, X: np.ndarray, y: np.ndarray, include_analysis=False) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_predict = self._predict(X)
        if include_analysis:
            self.loss_analysis(y, y_predict)
        return loss_functions.mean_square_error(y, y_predict)

    def loss_analysis(self, y_true, y_pred):
        pearsons_corr = (np.cov(y_pred, y_true)[0, 1] /
                         (np.std(y_pred) * np.std(y_true)))

        plot = go.Figure(data=[go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker_color='rgba(199, 10, 165, .9)')
        ])

        plot.update_layout(
            title="true vs pred " + str(pearsons_corr),
            xaxis_title="y_true",
            yaxis_title="y_pred",
        )
        plot.show()

    def fit_predict_loss(self, train_X, train_y, test_X, test_y, include_analysis=False):
        self._fit(train_X, train_y)
        return self._loss(test_X, test_y, include_analysis)
