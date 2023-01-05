import pandas as pd
from UtilsCreateDataFrame import *
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy_ext as npe
from statsmodels.tsa.stattools import adfuller

@dataclass
class ResultLR:
    beta: float
    intercept: float
    r2: float
    res_std: float
    res_mean: float
    stationarity_pvalue: float


def p_value_Stationary(x):
    test = adfuller(x)
    p_val = test[1]
    return p_val


def linearRegression_np(x, y):
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
    p_val = p_value_Stationary(residuals)

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
    return df_result