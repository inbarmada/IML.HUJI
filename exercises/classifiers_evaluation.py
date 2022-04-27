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
        # print(y)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        y_lda = lda.predict(X)
        print(lda.loss(X, y))
        print(lda.likelihood(X))
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        y_gnb = gnb.predict(X)
        print(gnb.loss(X, y))
        print(gnb.likelihood(X))
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
            x=X[:, 0], y=X[:, 1],
            marker=dict(color=y_gnb, colorscale="Viridis"),
            mode="markers", marker_symbol=y), row=1, col=2)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
