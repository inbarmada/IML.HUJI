import numpy as np
from typing import Callable, NoReturn

from IMLearn import BaseEstimator
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner
    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator
    self.iterations_: int
        Number of boosting iterations to perform
    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator
        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator
        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        num_samples = X.shape[0]
        num_features = X.shape[1]
        self.D_ = np.full(num_samples, 1.0/num_samples)
        self.models_ = [DecisionStump() for i in range(self.iterations_)]
        self.weights_ = np.zeros(self.iterations_)
        for i in range(self.iterations_):
            # Fit model and put into model array
            self.models_[i].fit(X, y * self.D_)
            y_pred = self.models_[i].predict(X)

            # calculate next D
            epsilon = np.sum(np.where(np.sign(y) == np.sign(y_pred), 0, self.D_))
            # print(i, epsilon, np.sum(np.where(y == y_pred, 0, 1)), self.D_, self.models_[i].loss(X, y), y_pred)
            # self.weights_[i] = 1
            # if epsilon != 0:
            self.weights_[i] = 1.0/2 * np.log((1.0 - epsilon) / epsilon)
            d_factor = np.exp(-self.weights_[i] * (y * y_pred))
            next_d = self.D_ * d_factor
            self.D_ = next_d / np.sum(next_d)


    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)


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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = np.zeros(X.shape[0])
        for i in range(T):
            y_pred += self.weights_[i] * self.models_[i].predict(X)
        return np.where(y_pred >= 0, 1, -1)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        T: int
            The number of classifiers (from 1,...,T) to be used for prediction
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred)
