from ModelFactory import ModelFactory

from IMLearn import BaseEstimator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import classification_report, mean_squared_error
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


from sklearn.preprocessing import MinMaxScaler

from challenge.Trainer import Trainer

pd.options.mode.chained_assignment = None

DATA_PATH = "../datasets/agoda_cancellation_train.csv"
START_DATE = '2018-07-07'
END_DATE = '2018-09-09'

def calc_date_diff(difference_name: str, category_one: str, category_two: str,
                   features):
    features.loc[:, difference_name] = \
        (pd.to_datetime(features.loc[:, category_one]) -
         pd.to_datetime(features.loc[:, category_two])) \
            .dt.total_seconds() / (60 * 60 * 24)


def split_date_time(category_name: str, column_name: str, features,
                    include_time=True):
    features[category_name + "_month"] = \
        pd.DatetimeIndex(features[column_name]).month

    features[category_name + "_day"] = \
        pd.DatetimeIndex(features[column_name]).day

    if include_time:
        features.loc[:, category_name + "_hour"] = \
            pd.DatetimeIndex(features[column_name]).hour


def get_num(x, d):
    try:
        if x[-1] == 'P':
            return int(x[:-1])
        else:
            return int(x[:-1]) * 100 / d
    except:
        return 0


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    indx_drop = full_data[full_data["checkin_date"] <= full_data["cancellation_datetime"]].index
    full_data.drop(indx_drop, inplace=True)

    indx_drop = full_data[pd.to_datetime(full_data["booking_datetime"]).dt.date >= pd.to_datetime(
        full_data["cancellation_datetime"])].index
    full_data.drop(indx_drop, inplace=True)

    full_data.reset_index(inplace=True)

    features = full_data[[
        # "hotel_id",
        "accommadation_type_name",
        "original_selling_amount",
        "booking_datetime",
        "checkin_date",
        "checkout_date",
        "charge_option",
        "original_payment_method",
        "original_payment_type",
        "hotel_star_rating",
        "hotel_country_code",
        "customer_nationality",
        "guest_is_not_the_customer",
        "guest_nationality_country_name",
        "no_of_adults",
        "no_of_children",
        "no_of_extra_bed",
        "no_of_room",
        "origin_country_code",
        "original_payment_currency",
        "is_user_logged_in",
        "request_nonesmoke",
        "request_latecheckin",
        "request_highfloor",
        "request_largebed",
        "request_twinbeds",
        "request_airport",
        "request_earlycheckin",
        "cancellation_policy_code",
        "h_customer_id"
    ]]

    process_dates(features)


    # Cancellation processing
    cancellation_policy_processing(features)
    # days_to_end_of_cancellation_week = (features["booking_datetime"] - pd.to_datetime('2020-12-13')).dt.days


    # Clean
    features.loc[:, "request_nonesmoke"] = features[
        "request_nonesmoke"].fillna(0)
    features.loc[:, "request_latecheckin"] = features[
        "request_latecheckin"].fillna(0)
    features.loc[:, "request_highfloor"] = features[
        "request_highfloor"].fillna(0)
    features.loc[:, "request_largebed"] = features["request_largebed"].fillna(
        0)
    features.loc[:, "request_twinbeds"] = features["request_twinbeds"].fillna(
        0)
    features.loc[:, "request_airport"] = features["request_airport"].fillna(0)
    features.loc[:, "request_earlycheckin"] = features[
        "request_earlycheckin"].fillna(0)
    features.loc[:, "is_user_logged_in"] = features[
        "is_user_logged_in"].astype(int)

    # Dummies
    dummies_columns = ["accommadation_type_name", "charge_option",
                       "original_payment_type", "original_payment_method",
                       "original_payment_currency", "hotel_country_code",
                       "customer_nationality",
                       "guest_nationality_country_name",
                       "origin_country_code", "h_customer_id"]

    for d in dummies_columns:
        top_values = features[d].value_counts()
        top_values = top_values[top_values > 100]
        features[d] = [x if x in top_values else np.nan for x in features[d]]

    features = pd.get_dummies(features, columns=dummies_columns)

    print(features.iloc[0])

    labels = ~full_data['cancellation_datetime'].isnull()

    features.drop("booking_datetime", inplace=True, axis=1)
    features.drop("checkin_date", inplace=True, axis=1)
    features.drop("checkout_date", inplace=True, axis=1)


    # Check correlation
    # correlation_heatmap(features, labels)


    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    data_frame_features = pd.DataFrame(scaled_features, columns=features.columns)

    regression_data = data_frame_features[labels]
    cancellation_date = pd.to_datetime(full_data["cancellation_datetime"]).dt.date

    booking_date = pd.to_datetime(full_data["booking_datetime"]).dt.date
    cancellation_date_regression = cancellation_date[labels]
    booking_date_regression = booking_date[labels]
    regression_labels = (cancellation_date_regression - booking_date_regression).dt.days

    checkein_date = pd.to_datetime(full_data["checkin_date"]).dt.date
    ##
    # regression_data["cancellation_minus_booking"] = (cancellation_date - booking_date).dt.days
    # regression_data["checkin_minus_cancellation"] = (checkein_date - cancellation_date).dt.days
    # regression_data = scaler.fit_transform(regression_data)
    ##
    data_frame_features["booking_date"] = booking_date
    # labels["cancellation_date"] = cancellation_date

    feature_columns = data_frame_features.columns
    return data_frame_features, labels, regression_data, regression_labels, cancellation_date, feature_columns


def load_test_data(filename: str, feature_columns):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename)

    features = full_data[[
        # "hotel_id",
        "accommadation_type_name",
        "original_selling_amount",
        "booking_datetime",
        "checkin_date",
        "checkout_date",
        "charge_option",
        "original_payment_method",
        "original_payment_type",
        "hotel_star_rating",
        "hotel_country_code",
        "customer_nationality",
        "guest_is_not_the_customer",
        "guest_nationality_country_name",
        "no_of_adults",
        "no_of_children",
        "no_of_extra_bed",
        "no_of_room",
        "origin_country_code",
        "original_payment_currency",
        "is_user_logged_in",
        "request_nonesmoke",
        "request_latecheckin",
        "request_highfloor",
        "request_largebed",
        "request_twinbeds",
        "request_airport",
        "request_earlycheckin",
        "cancellation_policy_code",
        "h_customer_id"
    ]]

    process_dates(features)


    # Cancellation processing
    cancellation_policy_processing(features)
    # days_to_end_of_cancellation_week = (features["booking_datetime"] - pd.to_datetime('2020-12-13')).dt.days


    # Clean
    features.loc[:, "request_nonesmoke"] = features[
        "request_nonesmoke"].fillna(0)
    features.loc[:, "request_latecheckin"] = features[
        "request_latecheckin"].fillna(0)
    features.loc[:, "request_highfloor"] = features[
        "request_highfloor"].fillna(0)
    features.loc[:, "request_largebed"] = features["request_largebed"].fillna(
        0)
    features.loc[:, "request_twinbeds"] = features["request_twinbeds"].fillna(
        0)
    features.loc[:, "request_airport"] = features["request_airport"].fillna(0)
    features.loc[:, "request_earlycheckin"] = features[
        "request_earlycheckin"].fillna(0)
    features.loc[:, "is_user_logged_in"] = features[
        "is_user_logged_in"].astype(int)

    # Dummies
    dummies_columns = ["accommadation_type_name", "charge_option",
                       "original_payment_type", "original_payment_method",
                       "original_payment_currency", "hotel_country_code",
                       "customer_nationality",
                       "guest_nationality_country_name",
                       "origin_country_code", "h_customer_id"]

    for d in dummies_columns:
        top_values = features[d].value_counts()
        top_values = top_values[top_values > 100]
        features[d] = [x if x in top_values else np.nan for x in features[d]]

    features = pd.get_dummies(features, columns=dummies_columns)

    print(features.iloc[0])

    features.drop("booking_datetime", inplace=True, axis=1)
    features.drop("checkin_date", inplace=True, axis=1)
    features.drop("checkout_date", inplace=True, axis=1)

    # features = features[features.columns.isin(feature_columns)]
    # Get missing columns in the training test
    missing_cols = set(feature_columns) - set(features.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        features[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    features = features[feature_columns]
    missing_cols = set(feature_columns) - set(features.columns)
    print('1', missing_cols)
    missing_cols = set(features.columns) - set(feature_columns)
    print('2', missing_cols)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    data_frame_features = pd.DataFrame(scaled_features, columns=features.columns)


    return data_frame_features



def correlation_heatmap(features, labels):
    check_corr = features.copy()
    check_corr['labels'] = labels
    corr = abs(check_corr.corr())
    print(corr)
    heatmap_plot = go.Figure(data=go.Heatmap(
        z=corr.values.tolist(),
        x=corr.columns.tolist(),
        y=corr.index.tolist()))
    heatmap_plot.update_layout(
        title="correlation",
        xaxis_title="columns",
        yaxis_title="columns"
    )
    heatmap_plot.show()


def cancellation_policy_processing(features):
    days_to_start_of_cancellation_week = [(pd.to_datetime('2018-12-07') - x)
                                          for x in pd.to_datetime(
            features["booking_datetime"])]
    days_to_end_of_cancellation_week = [(pd.to_datetime('2018-12-07') - x) for
                                        x in pd.to_datetime(
            features["booking_datetime"])]
    # print(days_to_start_of_cancellation_week)

    # check if there's a number x followed by a D in "cancellation_policy_code",
    # where x is between days_to_start_of_cancellation_week to
    # days_to_end_of_cancellation_week 
    # put result in a new column and then convert from boolean to int
    
    # example: days_to_start_of_cancellation_week = 5
    # days_to_end_of_cancellation_week = 12
    # look for a number x>=5 and x<=12 such that "xD" appears in policy code
    # - if the code is 10D75P -> return yes
    # - if the code is 20D2N_3N -> return no
    
    features["cancellation_policy_code"] = \
        features["cancellation_policy_code"].str.split('_').str[-1]
    features["cancellation_policy_days"] = \
        features["cancellation_policy_code"].str.split('D').str[0]
    features["cancellation_policy_days"] = [
        0.25 if x[-1] in ['P', 'N'] else int(x) for x in
        features["cancellation_policy_days"]]
    features["cancellation_policy_percents"] = \
        features["cancellation_policy_code"].str.split('D').str[-1]
    features["cancellation_policy_percents"] = features.apply(
        lambda x: get_num(x.cancellation_policy_percents,
                          x.checkin_to_checkout_days), axis=1)
    features["cancellation_policy_days_times_percents"] = \
        features["cancellation_policy_days"] * \
        features["cancellation_policy_percents"]
    features.drop("cancellation_policy_code", inplace=True, axis=1)


def process_dates(features):
    # Add date difference category
    calc_date_diff("booking_to_checkin_days", "checkin_date",
                   "booking_datetime", features)
    calc_date_diff("checkin_to_checkout_days", "checkout_date", "checkin_date",
                   features)

    # Split booking date time
    split_date_time("booking", "booking_datetime", features)
    split_date_time("checkin", "checkin_date", features, False)
    split_date_time("checkout", "checkout_date", features, False)


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)



def is_in_week(cancellation):
    # return (cancellation >= datetime.strptime('2018-12-07', '%Y-%m-%d')) & (
    #             cancellation <= datetime.strptime('2018-12-13', '%Y-%m-%d'))
    return (cancellation >= datetime.strptime(START_DATE, '%Y-%m-%d')) & (
            cancellation <= datetime.strptime(END_DATE, '%Y-%m-%d'))

def shift_dates_and_check_is_in_week(booking_date, days_to_cancellation):
    # cancellation = pd.to_datetime(booking_date) + days_to_cancellation
    # return cancellation.between('2018-12-07', '2018-12-13')
    cancellation = pd.to_datetime(booking_date) + timedelta(int(days_to_cancellation))
    return is_in_week(cancellation)

""" Make sense of the output prediction in the last stage """
def process_prediction(y_predict, booking_date_regression):
    for i in range(len(y_predict)):
        if not(y_predict[i]):
            y_predict[i] = 0
        else:
            y_predict[i] = max(y_predict[i], 0)
    # todo : check if for can be "apply"
    for i in range(len(y_predict)):
        y_predict[i] = shift_dates_and_check_is_in_week(booking_date_regression.iloc[i], y_predict[i])

    return y_predict


def evaluate_model(test_y, y_predict, label):
    print("********************** {} **********************".format(label))
    print(classification_report(test_y, y_predict))


# def train_model_two(regression_data, regression_labels, booking_date_regression, model_factory):
#
#     train_X, test_X, train_y, test_y = train_test_split(regression_data,
#                                                         regression_labels,
#                                                         test_size=0.25,
#                                                         random_state=42)
#     # Train model using Linear regression
#     model = model_factory.createModel("LinearReg")
#     model.fit(regression_data, regression_labels)
#     # Predict and classify if will cancel in bucket
#     y_predict = process_prediction(model.predict(test_X), booking_date_regression)
#
#     # evaluate_model(y_predict, test_y)


def train_model_one(train_X_one, train_y_one, model_one):
    # Train model one
    model_one.fit(train_X_one.drop(columns="booking_date"), train_y_one)

def train_model_two(train_X_two, train_y_two, model_two):
    # Train model two
    model_two.fit(train_X_two, train_y_two)

def predict_model_one(test_X_one, model_one):
    threshold = 0.38
    predicted_proba = model_one.predict_proba(
        test_X_one.drop(columns="booking_date"))
    pred_y_one = (predicted_proba[:, 1] >= threshold).astype('int')
    return pred_y_one

def predict_model_two (test_X_two, model_two):
    # Predict with model 2
    pred_y_two = model_two.predict(test_X_two)
    return pred_y_two

def evaluate_model_one(test_y_one, pred_y_one):
    evaluate_model(test_y_one, pred_y_one, "Model 1")

def evaluate_model_two(test_y_two, pred_y_two):
    print("MSE of model 2:", mean_squared_error(pred_y_two, test_y_two))


def full_prediction(data, model_one, model_two):
    # Predict
    pred_y_one = predict_model_one(data, model_one)
    pred_y_two = predict_model_two(
        data[pred_y_one == 1].drop(columns="booking_date"), model_two)

    # Check whether cancellation date prediction happen in relevant week
    cancellation_prediction = \
        process_prediction(pred_y_two, data['booking_date'])

    # Put the results here
    cancellation_results = np.zeros(len(pred_y_one))
    # In every booking that the first model cancelled,
    # write whether it was cancelled in the relevant week
    cancellation_prediction_index = 0
    for i in range(len(cancellation_results)):
        if cancellation_results[i]:
            # print(cancellation_prediction[i], cancellation_prediction[cancellation_prediction_index])
            cancellation_results[i] = \
                cancellation_prediction[cancellation_prediction_index]
            cancellation_prediction_index += 1
        else:
            cancellation_results[i] = 0

    # final_y = np.zeros(len(test_y_one))
    # for i in range(len(prediction_y)):
    #     if prediction_y[i]:
    #         final_y[i] = True if y[index] else False
    #         index += 1
    #     else:
    #         final_y[i] = False

    return cancellation_results


if __name__ == '__main__':
    np.random.seed(0)

    # Load data and organize to send to models for training and testing
    df, cancellation_labels, regression_data, regression_labels, cancellation_date, feature_columns = load_data(DATA_PATH)
    train_X_one, test_X_one, train_y_one, test_y_one = train_test_split(df, cancellation_labels, test_size=0.25, random_state=42)
    train_X_two, test_X_two, train_y_two, test_y_two = train_test_split(regression_data, regression_labels, test_size=0.25, random_state=42)

    test_data = load_test_data("../datasets/test_set_week_1.csv", feature_columns)


    model_factory = ModelFactory()

    # 1. Train model 1 to learn and predict which samples will cancel; obtain all such samples
    model_one = model_factory.createModel("RandForest", [800]).fit(train_X_one.drop(columns="booking_date"), train_y_one)
    # model_one = model_factory.createModel("LogisticReg").fit(train_X_one.drop(columns="booking_date"), train_y_one)

    ######
    threshold = 0.38

    predicted_proba = model_one.predict_proba(test_X_one.drop(columns="booking_date"))
    prediction_y = (predicted_proba[:, 1] >= threshold).astype('int')

    ######

    # prediction_y = model_one.predict(test_X_one.drop(columns="booking_date"))

    evaluate_model(test_y_one, prediction_y, "Model 1")
    indices_to_cancel = np.where(prediction_y == True)[0]
    relevant_indexes = test_X_one.iloc[indices_to_cancel].index.values.astype(int)
    predicted_samples_to_cancel = df.iloc[relevant_indexes].drop(columns="booking_date")
    # 2. Train model 2 to learn and predict whether a sample that cancels will cancel in the desired bucket
    # model_two = model_factory.createModel("LinearReg").fit(regression_data, regression_labels)

    model_two = model_factory.createModel("RandForestReg").fit(regression_data, regression_labels)


    # 3. Have model 2 predict whether those we believe will cancel, will indeed cancel in the desired bucket
    y = process_prediction(model_two.predict(predicted_samples_to_cancel), test_X_one.loc[relevant_indexes]["booking_date"])
    # 4. Process output and evaluate performance
    # for i in relevant_indexes:
    #     test_value = shift_dates_and_check_is_in_week(test_X_one.loc[i]["booking_date"], test_y_one.loc[i])
    #     test_y_one.iloc[i] = True if test_value else False
    final_y = np.zeros(len(test_y_one))
    index = 0
    # for i in range(len(prediction_y)):
    #     if prediction_y[i]:
    #         final_y[i] = True if y[index] else False
    #         index += 1
    #     else:
    #         final_y[i] = False

    for i in range(len(prediction_y)):
        if prediction_y[i]:
            final_y[i] = True if y[index] else False
            index += 1
        else:
            final_y[i] = False

    for i in test_y_one.index.values.astype(int):
        if test_y_one.loc[i]:
            test_y_one.loc[i] = shift_dates_and_check_is_in_week(cancellation_date.loc[i], 0)

    evaluate_model(test_y_one, final_y, "Final Results")
    test_cancellation_results = full_prediction(test_data, model_one, model_two)
    np.savetxt('results.csv', test_cancellation_results, delimiter=',')


#
# if __name__ == '__main__':
#     np.random.seed(0)
#     # Load data and organize to send to models for training and testing
#     df, cancellation_labels, regression_data, regression_labels, cancellation_date, feature_columns = load_data(DATA_PATH)
#
#     train_X_one, test_X_one, train_y_one, test_y_one = train_test_split(df, cancellation_labels, test_size=0.25, random_state=42)
#     train_X_two, test_X_two, train_y_two, test_y_two = train_test_split(regression_data, regression_labels, test_size=0.25, random_state=42)
#
#     model_factory = ModelFactory()
#
#     # Build model one
#     model_one = model_factory.createModel("RandForest", [750])
#
#     train_model_one(train_X_one, train_y_one, model_one)
#     pred_y_one = predict_model_one(test_X_one, model_one)
#     evaluate_model_one(test_y_one, pred_y_one)
#
#     # Build model two
#     model_two = model_factory.createModel("RandForestReg")
#
#     train_model_two(train_X_two, train_y_two, model_two)
#     pred_y_two = predict_model_two(test_X_two, model_two)
#     evaluate_model_two(test_y_two, pred_y_two)
#
#     # Full data prediction on test_X_one
#     test_cancellation_results = \
#         full_prediction(test_X_one, model_one, model_two)
#
#     # Get true results
#     indices_of_test_X_one = test_X_one.index.values.astype(int)
#     true_cancellation_results = cancellation_date.loc[indices_of_test_X_one]  # THIS IS THE WRONG data set
#     # SHOULD BE DATES MATCHING TEST_X_ONE
#     # ANyway anyone know how we get the dates of test_X_one
#     #no the features matching test_X_one (test_y_oneI'm join
#     for i in range(len(true_cancellation_results)):
#         if not (pd.isnull(cancellation_date.iloc[i])):
#             if (pd.to_datetime(START_DATE) <=
#                     pd.to_datetime(cancellation_date.iloc[i])
#                     <= pd.to_datetime(END_DATE)):
#                 true_cancellation_results.iloc[i] = 1
#             else:
#                 true_cancellation_results.iloc[i] = 0
#         else:
#             true_cancellation_results.iloc[i] = 0
#
#
#     # diff = test_cancellation_results - true_cancellation_results
#
#
#     # evaluate_model(true_cancellation_results, test_cancellation_results, "MODEL 1 + 2")
#     temp = [x for x in true_cancellation_results.reset_index()['cancellation_datetime']]
#     evaluate_model(temp, test_cancellation_results,
#                    "MODEL 1 + 2")
#     # print("Percent correct: ", 1 - (sum(abs(diff))/len(diff)))
#     # Save to csv file
#     np.savetxt('results.csv', test_cancellation_results, delimiter=',')
#
#     # # 1. Train model 1 to learn and predict which samples will cancel; obtain all such samples
#     # model_one = model_factory.createModel("RandForest", [750])
#     # model_one.fit(train_X_one.drop(columns="booking_date"), train_y_one)
#     # # model_one = model_factory.createModel("LogisticReg").fit(train_X_one.drop(columns="booking_date"), train_y_one)
#     #
#     # ######
#     # threshold = 0.38
#     #
#     # predicted_proba = model_one.predict_proba(test_X_one.drop(columns="booking_date"))
#     # prediction_y = (predicted_proba[:, 1] >= threshold).astype('int')
#     #
#     # ######
#     #
#     # # prediction_y = model_one.predict(test_X_one.drop(columns="booking_date"))
#     #
#     # evaluate_model(test_y_one, prediction_y, "Model 1")
#     #
#     # indices_to_cancel = np.where(prediction_y == True)[0]
#     # relevant_indexes = test_X_one.iloc[indices_to_cancel].index.values.astype(int)
#     # predicted_samples_to_cancel = df.iloc[relevant_indexes].drop(columns="booking_date")
#     # # 2. Train model 2 to learn and predict whether a sample that cancels will cancel in the desired bucket
#     # # model_two = model_factory.createModel("LinearReg").fit(regression_data, regression_labels)
#     #
#     # model_two = model_factory.createModel("RandForestReg")
#     # model_two.fit(regression_data, regression_labels)
#     #
#     #
#     # # 3. Have model 2 predict whether those we believe will cancel, will indeed cancel in the desired bucket
#     # y = process_prediction(model_two.predict(predicted_samples_to_cancel), test_X_one.loc[relevant_indexes]["booking_date"])
#     # # 4. Process output and evaluate performance
#     # # for i in relevant_indexes:
#     # #     test_value = shift_dates_and_check_is_in_week(test_X_one.loc[i]["booking_date"], test_y_one.loc[i])
#     # #     test_y_one.iloc[i] = True if test_value else False
#     # final_y = np.zeros(len(test_y_one))
#     # index = 0
#     # # for i in range(len(prediction_y)):
#     # #     if prediction_y[i]:
#     # #         final_y[i] = True if y[index] else False
#     # #         index += 1
#     # #     else:
#     # #         final_y[i] = False
#     #
#     # for i in range(len(prediction_y)):
#     #     if prediction_y[i]:
#     #         final_y[i] = True if y[index] else False
#     #         index += 1
#     #     else:
#     #         final_y[i] = False
#     #
#     # for i in test_y_one.index.values.astype(int):
#     #     if test_y_one.loc[i]:
#     #         test_y_one.loc[i] = shift_dates_and_check_is_in_week(cancellation_date.loc[i], 0)
#     #
#     # evaluate_model(test_y_one, final_y, "Final Results")
#     #
#     # # Get predictions for class data set
#     # test_data = load_test_data("../datasets/test_set_week_1.csv")
#     #
#     # # Run model 1
#     # predicted_proba = \
#     #     model_one.predict_proba(test_data.drop(columns="booking_date"))
#     # test_data_cancellation_prediction = \
#     #     (predicted_proba[:, 1] >= threshold).astype('int')
#     #
#     # # drop indices that are not predicted to cancel
#     # test_indices_to_cancel = np.where(test_data_cancellation_prediction)[0]
#     # test_relevant_indexes = \
#     #     test_data_cancellation_prediction.iloc[indices_to_cancel].index.values.astype(int)
#     # test_predicted_samples_to_cancel = test_data.iloc[test_relevant_indexes].drop(columns="booking_date")
#     #
#     # # Run model 2
#     # test_results_cancelled = process_prediction(model_two.predict(test_predicted_samples_to_cancel),
#     #                    test_data.loc[relevant_indexes]["booking_date"])
#     #
#     # # Print
#     # final_y = np.zeros(len(test_data))
#     # index = 0
#     # for i in range(len(prediction_y)):
#     #     if prediction_y[i]:
#     #         final_y[i] = True if test_results_cancelled[index] else False
#     #         index += 1
#     #     else:
#     #         final_y[i] = False
#     #
#     # for i in test_y_one.index.values.astype(int):
#     #     if test_y_one.loc[i]:
#     #         test_y_one.loc[i] = shift_dates_and_check_is_in_week(cancellation_date.loc[i], 0)
