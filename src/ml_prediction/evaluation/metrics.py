import logging
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def mape(y, yhat, perc=True):
    """
    Safe computation of the Mean Average Percentage Error

    Parameters
    ----------
    y: pd.Series or np.array holding the actual values
    yhat: pd.Series or np.array holding the predicted values
    perc: if True return the value in percentage

    Returns
    -------
    the MAPE value
    """
    # err = -1.0
    try:
        m = len(y.index) if type(y) == pd.Series else len(y)
        n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)
        assert m == n
        mape = []
        for a, f in zip(y, yhat):
            # avoid division by 0
            if f > 1e-6:
                mape.append(np.abs((a - f) / a))
        mape = np.mean(np.array(mape))
        return mape * 100.0 if perc else mape
    except AssertionError:
        logging.info(f"Wrong dimension for MAPE calculation: y = {m}, yhat = {n}")
        return -1.0


mape_scorer = make_scorer(mape, greater_is_better=False)


def symmetryc_mean_absolute_percentage_error(y_true, y_pred):

    """Return the MAPE metric from given both arrays of true and predicted values"""

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(2 * np.abs((y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))) * 100


def metrics_regressor(y_true, y_pred):
    """return rmse, mae, mad, mape"""

    df = pd.DataFrame(data={"True": y_true, "Predicted": y_pred}).dropna()
    df = df.dropna()

    R2 = round(r2_score(df["True"], df["Predicted"]), 3)
    mse = round(mean_squared_error(df["True"], df["Predicted"]), 3)
    rmse = round(sqrt(mean_squared_error(df["True"], df["Predicted"])), 3)
    mae = round(mean_absolute_error(df["True"], df["Predicted"]), 3)
    mad = round(median_absolute_error(df["True"], df["Predicted"]), 3)
    mape = round(mean_absolute_percentage_error(df["True"], df["Predicted"]), 3)
    smape = round(symmetryc_mean_absolute_percentage_error(df["True"], df["Predicted"]), 3)

    metric_dict = {
        "R2": R2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAD": mad,
        "MAPE (%)": mape,
        "SMAPE (%)": smape,
    }

    return metric_dict


def metrics_barplots(df, forecast_time, true_column, fraunhofer_column, GFT_column):
    """calcule, plot and return naive, fraunhofer, GFT regeression metrics"""

    string_start = str(df.index.min())[:-6] + "h"
    string_end = str(df.index.max())[:-6] + "h"

    display(Markdown(f"### Error relative to {true_column} fom {string_start} to {string_end}"))
    display(Markdown(""))

    " plot and returns metrics   (naive vs fraunhofer vs GFT)"
    df = df.copy()

    true_value = df[true_column]
    GFT_prediction = df[GFT_column]
    fraunhofer_prediction = df[fraunhofer_column]
    naive = df[true_column].shift(forecast_time)

    me = pd.DataFrame(index=["Dummy", "Fraunhofer", "GFT"], columns=["RMSE (kW)", "MAE (kW)", "MAD (kW)", "MAPE (%)"])
    me.loc["Dummy"] = metrics_regressor(true_value, naive)
    me.loc["Fraunhofer"] = metrics_regressor(true_value, fraunhofer_prediction)
    me.loc["GFT"] = metrics_regressor(true_value, GFT_prediction)

    # logging.info(me)

    fig, axs = plt.subplots(1, 4, figsize=(17, 4))

    # fig.suptitle(f'ERROR fom {string_start} to {string_end}', y=1.1)

    for i, metric in enumerate(me.columns):
        axs[i].bar(height=me[metric], x=me.index, color=["orange", "green", "navy"])
        axs[i].set_title(metric)
    plt.plot()

    return me
