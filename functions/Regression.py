import pandas as pd
from functions.UtilsCreateDataFrame import *
import sklearn
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy_ext as npe
from statsmodels.tsa.stattools import adfuller, zivot_andrews

@dataclass
class ResultLR:
    beta: float
    intercept: float
    r2: float
    res_std: float
    res_mean: float
    stationarity_pvalue: float


def p_value_Stationary(x, stat_test = "adfuller"):
    if stat_test == "adfuller":
        test = adfuller(x, autolag = "AIC")
        p_val = test[1]
    if stat_test == "zivot_andrews":
        test = zivot_andrews(x, autolag = "AIC")
        p_val = test[1]
    
    return p_val


def linearRegression_np(x, y, stat_test = "adfuller"):
    """
    function that fits a linear regression and return several objects:
    Inputs:
        -x: ndarray
        -y: ndarray

    Outputs:
        -results: ResultLR

    """
    # create x vector and y vector
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # fit the model
    mod = LinearRegression().fit(x, y)
    # predicted values
    pred = mod.predict(x).reshape(-1, 1)
    r2 = r2_score(y_true=y, y_pred=pred)
    residuals = y.flatten() - pred.flatten()
    res_std = np.std(residuals)
    res_mean = np.mean(residuals)
    p_val = p_value_Stationary(residuals, stat_test = stat_test)

    result = ResultLR(
        mod.coef_.item(), mod.intercept_.item(), r2, res_std, res_mean, p_val
    )
    return result


def ResultDataFrame(result, index_input):
    """
    function that trasform the result of the rolling operation to a dataframe:
    Inputs:
        -result: list
                list of np.nan and ResultLR objects
        -index_input: Pandas Index
                index of the input dataframe of the rolling operation

    Outputs:
        -df_result: pd.DataFrame

    """

    # index of non na values
    index = ~pd.isnull(result)
    # dict to use for na values
    na_dict = {x: np.nan for x in result[index[0]]}
    # number of na values
    len_na = np.sum(~index)
    na_dict_list = [na_dict for j in range(len_na)]
    df_result = pd.DataFrame.from_records(
        na_dict_list + [asdict(s) for s in result[index]]
    )
    df_result = df_result.set_index(index_input)
    df_result["date_est"] = df_result.index
    return df_result


def fromTimetoPlainIndex(window, frequency):
    """
    function to get number of observations given a time window and a frequency of the dataframe
    Inputs:
        -window: dict
                datetime arguments dict
        -frequency: dict
                datetime arguments dict 
    Output: 
        -n_obs: int 
    
    """
    win = dt.timedelta(**window)
    freq = dt.timedelta(**frequency)
    n_obs = win/freq
    return n_obs 




