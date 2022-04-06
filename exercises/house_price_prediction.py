from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates().dropna()
    full_data = full_data[full_data["price"] > 0]
    features = full_data[[#"date",
                          "bedrooms", "bathrooms", "floors",
                          "sqft_living", "sqft_lot",
                          "waterfront", "view",
                          #"condition",
                          "grade",
                          "sqft_above", "sqft_basement",
                          #"yr_built",
                          "yr_renovated",
                          "zipcode",  #"lat", "long",
                          "sqft_living15", "sqft_lot15"]]

    # Process date (dropped because not useful)
    # features.loc[:, "months_since_2014"] = \
    #     (pd.DatetimeIndex(features["date"]).year - 2014) * 12 + \
    #     pd.DatetimeIndex(features["date"]).month
    # features = features.drop(columns=["date"])

    features.loc[:, "floors"] = features.loc[:, "floors"].astype(int)

    # features["yr_built_or_renovated"] = features[["yr_built", "yr_renovated"]].max(axis=1)
    features = pd.get_dummies(data=features, columns=["zipcode"], drop_first=True)

    labels = full_data["price"]
    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    X_std_dev = X.std()
    y_std_dev = y.std()

    X_and_y = X.copy()
    X_and_y['labels'] = y

    cov = X_and_y.cov()['labels']
    pearsons_corr = (cov / X_std_dev) / y_std_dev

    for c in X.columns:
        print(c, ' corr ', pearsons_corr[c])
        plot = go.Figure(data=[go.Scatter(
            x=X[c],
            y=y,
            mode='markers',
            marker_color='rgba(199, 10, 165, .9)')
        ])

        plot.update_layout(
            title="Parsons correlation of " + c + " and price is : " +
                  str(pearsons_corr[c]),
            xaxis_title=c,
            yaxis_title="Price",
        )
        plot.show()



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lr = LinearRegression()

    loss = lr.fit_predict_loss(train_X.to_numpy(), train_y.to_numpy(),
                               test_X.to_numpy(), test_y.to_numpy(), True)

    p_loss_std = np.zeros(91)
    p_loss = np.zeros(91)
    for p in range(10, 101):
        total_loss = np.zeros(10)
        for i in range(10):
            # Get sample from training data
            sample_size = int(len(train_X) * p / 100)
            sample_indices = np.random.choice(len(train_X), sample_size,
                                              replace=False)
            sample_train_X = train_X.iloc[sample_indices].to_numpy()
            sample_train_y = train_y.iloc[sample_indices].to_numpy()

            # Predict and get loss
            loss = lr.fit_predict_loss(sample_train_X, sample_train_y,
                                       test_X.to_numpy(), test_y.to_numpy())
            total_loss[i] = loss
        p_loss[p - 10] = np.mean(total_loss)
        p_loss_std[p - 10] = np.std(total_loss)

    plot = go.Figure(data=[
        go.Scatter(x=list(range(10, 100)),y=p_loss,mode='lines+markers',
                   marker_color='rgba(199, 10, 165, .9)'),
        go.Scatter(x=list(range(10, 100)), y=p_loss - 2 * p_loss_std,
                   fill=None, mode="lines", line=dict(color="lightgrey"),
                   showlegend=False),
        go.Scatter(x=list(range(10, 100)), y=p_loss + 2 * p_loss_std,
                   fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                   showlegend=False)
        ]
    )

    plot.update_layout(
        title="Average Loss vs. Percent of Training Data Used",
        xaxis_title="Percent of Training Data Used",
        yaxis_title="Average Loss",
    )
    plot.show()
