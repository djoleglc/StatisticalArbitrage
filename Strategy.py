from dataclasses import dataclass, asdict
from typing import List
import pandas
import numpy
import pandas as pd
import datetime


@dataclass
class ResultStrategy:
    ret_: List
    ret_type: List
    enter_: List
    exit_: List


def transactionCost(df_, asset_name_1, asset_name_2, beta, t, fee=0.1 / 100):
    """
    function used to calculate transaction costs given a fee rate
    asset 1 is the y
    asset 2 is the x
    Inputs:
        -df_: pd.DataFrame
                dataframe containing the price for the two assets considered
        -asset_name_1: str
                name of the first asset used as y in the linear regression
        -asset_name_2: str
                name of the second asset used as x in the linear regression
        -beta: float
                linear regression coefficient
        -t: TimeStamp
                date used to assess which day to consider
        -fee: float
                fee rate
    Output:
        -feeToPay: float
                fee to pay at date t given the trade we aim to do
    """
    total_amount = df_.loc[t, asset_name_1] + beta * df_.loc[t, asset_name_2]
    feeToPay = total_amount * fee
    return feeToPay


def apply_twosigma(
    df_,
    asset_name_1,
    asset_name_2,
    beta,
    intercept,
    sigma,
    start_date,
    end_date,
    fee_rate=0.1 / 100,
):
    """
    function to apply a simple two sigma strategy
    index of df must be datetime
    version with multiple trades in a trading period
    asset_one is the y in the regression
    asset_two is the x in the regression
    Inputs:
        -df_: pd.DataFrame
                dataframe containing the price of the two assets and with time index
        -asset_name_1: str
                name of the first asset used as y in the linear regression
        -asset_name_2: str
                name of the second asset used as x in the linear regression
        -beta: float
                linear regression coefficient
        -intercept: float
                linear regression intercept coefficient
        -sigma: float
                standard deviation of the linear regression residuals
        -start_date: TimeStamp
                first day of the trading period
        -end_date: TimeStamp
                last day of the trading period
        -fee_rate: float
                fee applied for each trade expressed as a percentage of the value of the trade
    Output:
        -result: ResultStrategy
                object containing the return of each trade, the type of each trade, the enter and the exit date
    """
    if end_date <= df_.index[-1]:
        df__ = df_.loc[start_date:end_date, :]
    else:
        df__ = df.loc[start_date:, :]

    df = pd.DataFrame(
        df__.loc[:, asset_name_1] - beta * df__.loc[:, asset_name_2],
        columns=["spread"],
        index=df__.index,
    )

    state = 0  # initial state
    result = ResultStrategy([], [], [], [])
    isFinaltime = lambda j: (j == len(df) - 2)

    for j, t in enumerate(df.index[:-1]):
        if state == 0:
            if df.loc[t, "spread"] > intercept + (2 * sigma):
                # here we SHORT
                state = -1
                enter_pos_date = df.index[j + 1]
                result.enter_.append(enter_pos_date)

            if df.loc[t, "spread"] < intercept + (-2 * sigma):
                # here we go LONG
                state = 1
                enter_pos_date = df.index[j + 1]
                result.enter_.append(enter_pos_date)

        elif (state == -1 and df.loc[t, "spread"] <= intercept) or (
            state == -1 and isFinaltime(j)
        ):
            state = 0
            t_ = df.index[j + 1]
            fee_cost = transactionCost(
                df__, asset_name_1, asset_name_2, beta, t_, fee=fee_rate
            )
            return_ = (
                (-fee_cost / df.loc[enter_pos_date, "spread"])
                - (df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"])
                / df.loc[enter_pos_date, "spread"]
                + 1
            )
            result.ret_.append(return_)
            result.ret_type.append("Short")
            result.exit_.append(t_)

        elif (state == 1 and df.loc[t, "spread"] >= intercept) or (
            state == 1 and isFinaltime(j)
        ):
            state = 0
            t_ = df.index[j + 1]
            fee_cost = transactionCost(
                df__, asset_name_1, asset_name_2, beta, t_, fee=fee_rate
            )
            return_ = (
                (-fee_cost / df.loc[enter_pos_date, "spread"])
                + (df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"])
                / df.loc[enter_pos_date, "spread"]
                + 1
            )
            result.ret_.append(return_)
            result.exit_.append(t_)
            result.ret_type.append("Long")

    return result


def applyStrategyRolling(df_beta, df_price, trading_window_day=1, thr_pval=0.10):
    """
    function that applies the two sigma strategy
    Inputs:
        -df_beta: pd.DataFrame
                dataframe containing the beta, the intercept, the residuals mean and standard deviation, the r2 and the p value connected to the adfuller test from the linear regression
        -df_price: pd.DataFrame
                dataframe containing the price of the two assets considered
        -trading_window_day: float
                length of the trading window expressed in day
        -thr_pval: float
                threshold for the adfuller test pvalue used to assess if a signal can be used
    Outputs:
        -stratResult: list
                list containing ResultStrategy objects
        -decision_trading_day: list
                list containing information about the day in which a signal is observed and used for trading

    """
    y_name = df_price.columns[1]
    x_name = df_price.columns[0]
    stratResult = []
    decision_trading_day = []
    # used only to initialize the for
    end_date = df_beta.index[0] - datetime.timedelta(days=1)
    last_start_date = df_beta.index[-2] - datetime.timedelta(days=trading_window_day)

    for j, t in enumerate(df_beta.index):
        row = df_beta.loc[t, :]
        if row.stationarity_pvalue <= thr_pval and t > end_date:
            decision_trading_day.append(row)
            init_date = df_beta.index[j + 1]
            end_date = init_date + datetime.timedelta(days=trading_window_day)

            strat = apply_twosigma(
                df_price,
                y_name,
                x_name,
                beta=row.beta,
                intercept=row.intercept,
                sigma=row.res_std,
                start_date=init_date,
                end_date=end_date,
            )
            stratResult.append(strat)

    return stratResult, decision_trading_day


def strategyDataFrame(result_strategy):
    """
    function to trasform the result of a trading strategy into a dataframe
    Input:
        -result_strategy: list
                list containing the first position a list of ResultStrategy objects and in the second position a list
                of rows extracted from a beta dataframes
    Output:
        -df_trades: pd.DataFrame
    """
    to_dataframe = []

    for index, elem in enumerate(zip(result_strategy[0], result_strategy[1])):
        strat, beta = elem[0], elem[1]
        b = beta.to_dict()
        for key, val in b.items():
            b[key] = [val] * len(strat.ret_)

        dict_ = asdict(strat)
        dict_.update(b)
        to_dataframe += [dict(zip(dict_, elem)) for elem in zip(*dict_.values())]

    df_trades = pd.DataFrame(to_dataframe)
    return df_trades
