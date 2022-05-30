import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    ada = AdaBoost(DecisionStump, 250)
    ada.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_loss_vec = [ada.partial_loss(train_X, train_y, i) for i in range(250)]
    test_loss_vec = [ada.partial_loss(test_X, test_y, i) for i in range(250)]

    pdf_plot = go.Figure()
    pdf_plot.add_trace(go.Scatter(
        name="train error",
        x=list(range(250)),
        y=train_loss_vec,
        mode='lines+markers')
    )

    pdf_plot.add_trace(go.Scatter(
        name="test error",
        x=list(range(250)),
        y=test_loss_vec,
        mode='lines+markers')
    )

    pdf_plot.update_layout(
        title="Adaboost error",
        xaxis_title="Number of learners",
        yaxis_title="Miscalculation error"
    )
    pdf_plot.show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    def get_t_partial_predict(t):
        def t_partial_predict(X):
            return ada.partial_predict(X, t)
        return t_partial_predict

    def create_plot(k, plot_title="", marker_size=5, train=False):
        if plot_title == "":
            plot_title = "Decision Surface for " + str(k) + " Learners"
        X = train_X if train else test_X
        y = train_y if train else test_y
        go.Figure(([decision_surface(get_t_partial_predict(k), *lims),
                       go.Scatter(x=X[:, 0], y=X[:, 1],
                                  mode='markers',
                                  marker=dict(color=y,
                                              colorscale=custom,
                                              size=marker_size),
                                  showlegend=False)]),
                  layout=go.Layout(title=plot_title)).show()

    create_plot(5)
    create_plot(50)
    create_plot(100)
    create_plot(250)
    # Question 3: Decision surface of best performing ensemble
    best_ensemble = np.argmin(test_loss_vec)

    title = "Ensemble of size " + str(best_ensemble) + " and accuracy: " + \
            str(1 - test_loss_vec[best_ensemble])
    create_plot(best_ensemble, title)
    # Question 4: Decision surface with weighted samples
    D = ada.D_ / (np.max(ada.D_)) * 5
    title = "Training Set with Weights"
    create_plot(250, title, D, True)




if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
