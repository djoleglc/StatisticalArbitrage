from dataclasses import dataclass, asdict
from typing import List
import pandas
import numpy
import pandas as pd
import datetime
from functions.Regression import *


@dataclass
class ResultStrategy:
    ret_: List
    ret_no_fee: List
    ret_type: List
    spread_enter: List
    spread_exit: List
    fee_enter: List
    fee_exit: List
    borrow_fee: List
    enter_: List
    exit_: List

        
def create_beta_table(coin_df, asset_name_1, asset_name_2, calibration_window, frequency = {"minutes" = 1},  safe_output_csv = False):
    """
    Input:
        - coinf_df : pd.DataFrame
                DataFrame with the prices of assets (coins)
        - asset_name_1 : str
                y variable
        - asset_name_2 : str
                x variable
        - window : int
                amount of days which determines the calibration window size
        - safe_output_csv : bool
                determines whether the output table should be saved as csv
    Output:
        - df_beta : pd.DataFrame
               dataframe containing beta, intercept, r2, res_std, res_mean, stationarity_pvalue, date_est
    """
    window = int(fromTimetoPlainIndex(window = calibration_window, frequency = frequency))
    index_input = coin_df.index
    y, x = coin_df[asset_name_1].to_numpy(), coin_df[asset_name_2].to_numpy(), 
    rolling = npe.rolling_apply(linearRegression_np, window, x, y, n_jobs=4)
    df_beta = ResultDataFrame(rolling, index_input[:last])
    if safe_output_csv:
        df_beta.to_csv(f"./df_beta_{asset_name_2}_{asset_name_1}_{calibration_window_days}_days.csv.gz")
    return df_beta




def getCombRet(coin_df, asset_name_1, asset_name_2, calib_trading_windows, p_values, stop_loss = 0.2, safe_beta_csv = False):
    """
    Input:
        - coinf_df : pd.DataFrame
                DataFrame with the prices of assets (coins)
        - asset_name_1 : str
                y variable
        - asset_name_2 : str
                x variable
        - calib_trading_windows : List
                list containing amount of days which determine the calibration/trading window size
        - p_values : List
                list containing the relevant p-values as a threshold for trading
        - stop_loss : float in the interval of (0,1)
                determines the maximum loss we are willing to take before exiting the position
        - safe_output_csv : bool
                determines whether the output table should be saved as csv
    Output:
        - ret_dict : Dictionary
               dictionary with keys = date (e.g. "2021-01") and the respective dataframe of all 
               combinations of p_values and calib_trading_windows
    """
    coin_df["Month"] = coin_df.index.to_period('M')
    ret_dict = {}
    for date in coin_df["Month"].unique():
        ret_dict[str(date)] = pd.DataFrame(columns = p_values, index = calib_trading_windows)
    
    for window in calib_trading_windows:
        try:
            beta_df = pd.read_csv(f"./data/df_beta_{asset_name_2}_{asset_name_1}.csv.gz") #_{window}_days
            beta_df = beta_df.set_index("time")
            beta_df.index = pd.to_datetime(beta_df.index)
            beta_df["Month"] = beta_df.index.to_period('M')
        except:
            beta_df = create_beta_table(coin_df, asset_name_1, asset_name_2, window, safe_beta_csv)
            beta_df = beta_df.set_index("time")
            beta_df.index = pd.to_datetime(beta_df.index)
            beta_df["Month"] = beta_df.index.to_period('M')
        
        trading_window = {"days": window, "hours": 0, "minutes": 0}
        
        for date in coin_df["Month"].unique():
            p_dict = {}
            for p_val in p_values:
                result_strategy = applyStrategyRolling(df_beta = beta_df.loc[beta_df.Month == date], 
                                                       df_price = coin_df.loc[coin_df.Month == date], 
                                                       trading_window = trading_window, 
                                                       thr_pval = p_val, 
                                                       quantile = 2, 
                                                       fee_rate = 0.1 / 100, 
                                                       stop_loss = stop_loss)
                df_trades = strategyDataFrame(result_strategy)
                
                total_ret = df_trades["ret_"].prod()
                p_dict[p_val] = total_ret
                
            helper_df = ret_dict[str(date)]
            helper_df.loc[window] = (p_dict)
            ret_dict[str(date)] = helper_df
    
    return ret_dict



borrowCost_dict = {
    """
    daily margin borrow interest from Binance
    """
    
    "ADAU" : 0.024500/100,
    "BNB"  : 0.300000/100,
    "BTC"  : 0.005699/100,
    "DOGE" : 0.023200/100,
    "ETH"  : 0.005699/100,
    "SOL"  : 0.109589/100,
    "XRP"  : 0.017000/100
}
        
def borrowCost(df_, short_asset_name, enter_date, exit_date, beta = 1):
    """
    function used to calculate the borrowing costs when shorting an asset
    Inputs:
        - df_: pd.DataFrame
                dataframe containing the price for the two assets considered
        - short_asset_name: str
                name of the asset which is shorted
        - enter_date: TimeStamp
                date when short position was entered
        - exit_date: TimeStamp
                date when short positioned was exited
        - beta: float
                amount of assets which are shorted (by default, only one asset is shorted)
    Output:
        - feeToPay: float
                borrowing fee to pay
    """
    daily_interest = borrowCost_dict[short_asset_name]
    time_delta = exit_date - enter_date
    hours_borrowed = numpy.ceil(time_delta.total_seconds() / 60 / 60)
    amount_borrowed = df_.loc[enter_date, short_asset_name]
    return daily_interest / 24 * hours_borrowed * amount_borrowed


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
    quantile=2,
    stop_loss = 0.2
):
    """
    function to apply a simple two sigma strategy
    index of df must be datetime
    version with multiple trades in a trading period
    asset_one is the y in the regression
    asset_two is the x in the regression
    Inputs:
        - df_ : pd.DataFrame
                dataframe containing the price of the two assets and with time index
        - asset_name_1 : str
                name of the first asset used as y in the linear regression
        - asset_name_2 : str
                name of the second asset used as x in the linear regression
        - beta : float
                linear regression coefficient
        - intercept : float
                linear regression intercept coefficient
        - sigma : float
                standard deviation of the linear regression residuals
        - start_date : TimeStamp
                first day of the trading period
        - end_date : TimeStamp
                last day of the trading period
        - fee_rate : float
                fee applied for each trade expressed as a percentage of the value of the trade
        - quantile : float
                two sigma uses quantile = 2 but this can be changed to an arbitrarily quantile
        - stop_loss : float in the interval of (0,1)
                determines the maximum loss we are willing to take before exiting the position
    Output:
        -result: ResultStrategy
                object containing the return of each trade, the type of each trade, the enter and the exit date
    """
    if end_date <= df_.index[-1]:
        df__ = df_.loc[start_date:end_date, :]
    else:
        df__ = df_.loc[start_date:, :]

    df = pd.DataFrame(
        df__.loc[:, asset_name_1] - beta * df__.loc[:, asset_name_2],
        columns=["spread"],
        index=df__.index,
    )

    state = 0  # initial state
    result = ResultStrategy([], [], [], [], [], [], [], [], [], [])
    isFinaltime = lambda j: (j == len(df) - 2)

    for j, t in enumerate(df.index[:-1]):
        if state == 0:
            if df.loc[t, "spread"] > intercept + (quantile * sigma):
                # here we SHORT
                state = -1
                enter_pos_date = df.index[j + 1]
                fee_enter = transactionCost(
                    df__,
                    asset_name_1,
                    asset_name_2,
                    beta,
                    df.index[j + 1],
                    fee=fee_rate,
                )
                result.fee_enter.append(fee_enter)
                result.spread_enter.append(df.loc[df.index[j + 1], "spread"])
                result.enter_.append(enter_pos_date)

            if df.loc[t, "spread"] < intercept + (-quantile * sigma):
                # here we go LONG
                state = 1
                enter_pos_date = df.index[j + 1]
                fee_enter = transactionCost(
                    df__,
                    asset_name_1,
                    asset_name_2,
                    beta,
                    df.index[j + 1],
                    fee=fee_rate,
                )
                result.fee_enter.append(fee_enter)
                result.spread_enter.append(df.loc[df.index[j + 1], "spread"])
                result.enter_.append(enter_pos_date)

        
        elif state == -1:
            t_ = df.index[j + 1]
            
            borrow_fee = borrowCost(
                    df__,
                    asset_name_1,
                    enter_pos_date,
                    t_
                )
            
            fee_exit = transactionCost(
                df__, asset_name_1, asset_name_2, beta, t_, fee=fee_rate
            )

            return_no_fee = (
                -(df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"])
                / df.loc[enter_pos_date, "spread"]
                + 1
            )

            return_ = (
                (-borrow_fee / df.loc[enter_pos_date, "spread"])
                + (-fee_enter / df.loc[enter_pos_date, "spread"])
                + (-fee_exit / df.loc[enter_pos_date, "spread"])
                + return_no_fee
            )
            
            if df.loc[t, "spread"] <= intercept or isFinaltime(j) or return_ <= 1 - stop_loss:
                state = 0
                
                if return_ <= 1 - stop_loss:
                    result.ret_.append(1 - stop_loss)
                else:
                    result.ret_.append(return_)

                result.borrow_fee.append(borrow_fee)
                result.fee_exit.append(fee_exit)
                result.ret_no_fee.append(return_no_fee)
                result.spread_exit.append(df.loc[t_, "spread"])
                result.ret_type.append("Short")
                result.exit_.append(t_)

        elif state == 1:
            t_ = df.index[j + 1]
            
            borrow_fee = borrowCost(
                    df__,
                    asset_name_2,
                    enter_pos_date,
                    t_,
                    beta
                )
            
            fee_exit = transactionCost(
                df__, asset_name_1, asset_name_2, beta, t_, fee=fee_rate
            )

            return_no_fee = (
                df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"]
            ) / df.loc[enter_pos_date, "spread"] + 1
            
            return_ = (
                (-borrow_fee / df.loc[enter_pos_date, "spread"])
                + (-fee_enter / df.loc[enter_pos_date, "spread"])
                + (-fee_exit / df.loc[enter_pos_date, "spread"])
                + return_no_fee
            )
            
            if df.loc[t, "spread"] >= intercept or isFinaltime(j) or return_ <= 1 - stop_loss:
                state = 0
                
                if return_ <= 1 - stop_loss:
                    result.ret_.append(1 - stop_loss)
                else:
                    result.ret_.append(return_)
                    
                result.borrow_fee.append(borrow_fee)
                result.fee_exit.append(fee_exit)
                result.spread_exit.append(df.loc[t_, "spread"])
                result.exit_.append(t_)
                result.ret_no_fee.append(return_no_fee)
                result.ret_type.append("Long")

    return result


def applyStrategyRolling(
    df_beta, df_price, trading_window, thr_pval=0.10, quantile=2, fee_rate=0.1 / 100, stop_loss = 0.2
):
    """
    function that applies the two sigma strategy
    Inputs:
        -df_beta: pd.DataFrame
                dataframe containing the beta, the intercept, the residuals mean and standard deviation, the r2 and the p value connected to the adfuller test from the linear regression
        -df_price: pd.DataFrame
                dataframe containing the price of the two assets considered
        -trading_window: dict
                dictionary containing inputs for datetime.timedelta, e.g {"hours":1, "minutes": 50..}
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
    delta_time = datetime.timedelta(**trading_window)
    last_start_date = df_beta.index[-2] - delta_time

    for j, t in enumerate(df_beta.index):
        row = df_beta.loc[t, :]
        if row.stationarity_pvalue <= thr_pval and t > end_date:
            decision_trading_day.append(row)
            init_date = df_beta.index[j + 1]
            end_date = init_date + delta_time

            strat = apply_twosigma(
                df_price,
                y_name,
                x_name,
                beta=row.beta,
                intercept=row.intercept,
                sigma=row.res_std,
                start_date=init_date,
                end_date=end_date,
                fee_rate=fee_rate,
                quantile=quantile,
                stop_loss = stop_loss
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
