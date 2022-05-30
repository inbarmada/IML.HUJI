from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        loss_vec = np.zeros((np.shape(X)[1], 2))
        thresh_vec = np.zeros((np.shape(X)[1], 2))
        for i in range(np.shape(X)[1]):
            thresh_vec[i, 0], loss_vec[i, 0] = self._find_threshold(X[:, i], y, -1)
            thresh_vec[i, 1], loss_vec[i, 1] = self._find_threshold(X[:, i], y, 1)
            # print(i, thresh_vec[i], loss_vec[i, 0], loss_vec[i, 1])

        # self.j_ = np.argmin(loss_vec) % (np.shape(X)[1])
        # sign = int(np.argmin(loss_vec) / (np.shape(X)[1]))
        sign = np.argmin(loss_vec) % (np.shape(X)[1])
        self.j_ = int(np.argmin(loss_vec) / (np.shape(X)[1]))
        self.sign_ = (2 * sign) - 1
        self.threshold_ = thresh_vec[int(self.j_), int(sign)]


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.i_t_predict(X[:, self.j_], self.threshold_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        num_vals = values.shape[0]
        loss_vec = np.zeros(num_vals)
        for t in range(num_vals):
            loss_vec[t] = self.i_t_loss(values, labels, values[t], sign)
            # print(loss_vec[t])
        return values[np.argmin(loss_vec)], loss_vec[np.argmin(loss_vec)]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.i_t_loss(X[:, self.j_], y, self.threshold_, self.sign_)


    def i_t_loss(self, X_col, y, t, sign):
        y_pred = self.i_t_predict(X_col, t, sign)
        return np.sum(np.where(np.sign(y) != np.sign(y_pred), np.absolute(y), 0))

    def i_t_predict(self, X_col, t, sign):
        y_pred = np.where(X_col >= t, sign, -sign)
        return y_pred

