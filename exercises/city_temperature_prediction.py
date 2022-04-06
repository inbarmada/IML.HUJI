from IMLearn.learners.regressors import LinearRegression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> (pd.DataFrame, pd.Series):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]). \
        drop_duplicates().dropna()
    full_data = full_data[full_data["Temp"] > -70]
    features = full_data[["Country",
                          "City",
                          # "Date",
                          "Year",
                          "Month",
                          # "Day",
                          # "Temp"
                          ]]
    features["DayOfYear"] = full_data["Date"].dt.dayofyear
    # features = pd.get_dummies(data=features, columns=["Country"], drop_first=True)
    # features = pd.get_dummies(data=features, columns=["City"], drop_first=True)

    labels = full_data["Temp"]

    return features, labels


def explore_data_for_country(X, y, country):
    X_Israel, y_Israel = get_samples_for_country(X, y, country)
    X_Israel["Temp"] = y_Israel
    X_Israel["Year"] = X_Israel["Year"].map(str)
    fig = px.scatter(X_Israel, x="DayOfYear", y="Temp", color="Year")
    fig.show()

    X_Israel["Temp"] = y_Israel
    temp_std_by_month = X_Israel.groupby("Month").agg('std')["Temp"]
    plot = go.Figure(data=[go.Bar(x=list(range(12)), y=temp_std_by_month)])
    plot.update_layout(
        title="Month vs. Standard Deviation of Temperature in Israel",
        xaxis_title="Month",
        yaxis_title="Standard Deviation of Temperature",
    )
    plot.show()


def get_samples_for_country(X, y, country):
    X_c = X[X["Country"] == country]
    y_c = y[X["Country"] == country]
    return X_c, y_c


def explore_data_all_countries(X, y):
    df = X.copy()
    df["Temp"] = y.copy()
    df_std_dev = df.groupby(["Country", "Month"]).std().reset_index()
    df = df.groupby(["Country", "Month"]).mean().reset_index()
    fig = px.line(df, x="Month", y="Temp", error_y=df_std_dev["Temp"],
                  color="Country")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data("..\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    explore_data_for_country(X, y, "Israel")

    # Question 3 - Exploring differences between countries
    explore_data_all_countries(X, y)

    # Question 4 - Fitting model for different values of `k`
    # Get samples from Israel
    X_Israel, y_Israel = get_samples_for_country(X, y, "Israel")

    # Split to train and test
    train_X, train_y, test_X, test_y = split_train_test(X_Israel, y_Israel)
    train_X = train_X["DayOfYear"]
    test_X = test_X["DayOfYear"]

    # Fit regression
    loss_k = np.zeros(10)
    for k in range(10):
        # Create and train model
        pr = PolynomialFitting(k + 1)
        pr.fit(train_X.to_numpy(), train_y.to_numpy())

        # Calculate loss
        loss = pr.loss(test_X.to_numpy(), test_y.to_numpy())

        # Print and save loss
        print(k + 1, ": ", round(loss, 2))
        loss_k[k] = round(loss, 2)

    # Draw loss graph
    plot = go.Figure(data=[go.Bar(x=list(range(1, 11)), y=loss_k)])
    plot.update_layout(
        title="Loss vs. Degree for Polynomial Fit",
        xaxis_title="Degree for Polynomial Fit",
        yaxis_title="Loss",
    )
    plot.show()

    # Question 5 - Evaluating fitted model on different countries
    # Fit model
    pr = PolynomialFitting(5)
    pr.fit(X_Israel["DayOfYear"].to_numpy(), y_Israel.to_numpy())

    # Check loss for each country
    Countries = ["The Netherlands", "South Africa", "Jordan"]
    loss_countries = np.zeros(len(Countries))
    for i in range(len(Countries)):
        # Get samples for country c
        X_c, y_c = get_samples_for_country(X, y, Countries[i])
        X_c = X_c["DayOfYear"]
        
        # Calculate loss
        loss_countries[i] = pr.loss(X_c.to_numpy(), y_c.to_numpy())

    # Create graph
    plot = go.Figure(data=[go.Bar(x=Countries, y=loss_countries)])
    plot.update_layout(
        title="Countries vs. Loss",
        xaxis_title="Countries",
        yaxis_title="Loss",
    )
    plot.show()

