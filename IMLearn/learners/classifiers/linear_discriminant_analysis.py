from typing import NoReturn

from .. import MultivariateGaussian
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))
        self.pi_ = np.zeros(len(self.classes_))

        X_minus_mu = X.copy()
        for i in range(len(self.classes_)):
            X_i = X[y == self.classes_[i]]
            self.mu_[i] = np.mean(X_i, axis=0)
            self.pi_[i] = X_i.shape[0] / X.shape[0]
            X_minus_mu[y == self.classes_[i]] -= self.mu_[i]

        self.cov_ = np.matmul(np.transpose(X_minus_mu), X_minus_mu) / \
                    X.shape[0]
        self._cov_inv = np.linalg.inv(self.cov_)

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
        A = np.matmul(self._cov_inv, np.transpose(self.mu_))
        muk_t_covinv_muk = np.sum(np.multiply(np.transpose(self.mu_),
                           np.matmul(self._cov_inv, np.transpose(self.mu_))),
                                  axis=0) / 2
        B = np.log(self.pi_) - muk_t_covinv_muk

        y = self.classes_[np.argmax(np.matmul(X, A) + B, axis=1)]
        return y

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            mu = self.mu_[i]
            factor = (1 / np.sqrt((2 * np.pi) ** len(mu) * np.linalg.det(self.cov_)))
            sigma_mult_x = np.matmul(self._cov_inv, np.transpose(X - mu))

            # pdfs = np.exp(-1 / (2 * np.matmul(np.matmul(np.transpose(X - mu), np.linalg.inv(cov))), (X - mu)))
            X_sigma_x = np.multiply(np.transpose(X - mu), sigma_mult_x)
            X_sigma_x = np.mean(X_sigma_x, axis=0)
            pdfs = np.exp(-1 / 2 * X_sigma_x)

            likelihood[:, i] = pdfs * factor

        return likelihood


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
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
