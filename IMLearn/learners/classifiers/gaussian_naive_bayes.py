from typing import NoReturn

from .. import UnivariateGaussian
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.cov_ = np.zeros((len(self.classes_), X.shape[1]))
        for k in range(len(self.classes_)):
            X_k = X[y == self.classes_[k]]
            self.mu_[k] = np.mean(X_k, axis=0)
            self.pi_[k] = X_k.shape[0] / X.shape[0]
            X_k_diff_squared = np.power(X_k - self.mu_[k], 2)
            self.cov_[k] = np.sum(X_k_diff_squared, axis=0) / \
                           (X_k_diff_squared.shape[0] - 1)

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
        y = self.classes_[np.argmax(self.likelihood(X), axis=1)]
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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            cov = self.cov_[i]
            mu = self.mu_[i]

            factor = (1 / np.sqrt((2 * np.pi) ** len(mu) * np.prod(cov)))
            first_product = np.matmul(np.power(X-mu, 2), 1 / cov)
            pdfs = np.exp(-1 / 2 * first_product)

            likelihood[:, i] = pdfs * factor * self.pi_[i]
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
