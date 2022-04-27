from math import atan2, pi

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load("../datasets/" + filename)
    return data[:, :2], data[:, 2]

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, response = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, response))

        p = Perceptron(True, 1000, callback)
        p.fit(X, response)

        # Plot figure
        plot = go.Figure(data=[go.Scatter(x=list(range(1, len(losses) + 1)),
                                          y=losses, mode='lines')])

        plot.update_layout(title="Loss vs. Iteration: " + n,
                           xaxis_title="Iteration", yaxis_title="Loss")
        plot.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        y_lda = lda.predict(X)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        y_gnb = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # from IMLearn.metrics import accuracy

        fig = make_subplots(rows=1, cols=2, start_cell="bottom-left",
              subplot_titles=("LDA: Accuracy " + str(1 - lda.loss(X, y)),
                              "GNB: Accuracy " + str(1 - gnb.loss(X, y))))
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            marker=dict(color=y_lda, colorscale="Viridis"),
            mode="markers", marker_symbol=y), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=lda.mu_[:, 0], y=lda.mu_[:, 1],
            mode="markers", marker_symbol=4, marker_size=10,
            marker_color="black"), row=1, col=1)
        for i in range(3):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=1)


        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            marker=dict(color=y_gnb, colorscale="Viridis"),
            mode="markers", marker_symbol=y), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=gnb.mu_[:, 0], y=gnb.mu_[:, 1],
            mode="markers", marker_symbol=4, marker_size=10,
            marker_color="black"), row=1, col=2)
        for i in range(3):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.cov_[i])), row=1, col=2)

        fig.update_layout(showlegend=False, title=f + " Graphs")
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
