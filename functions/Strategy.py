from dataclasses import dataclass, asdict
from typing import List
import datetime
import pandas
import numpy
from functions.UtilsGoogleDrive import *
import os
import pandas as pd
import datetime
from functions.Regression import *
import warnings
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from functions.UtilsRetrieveData import *
from functions.UtilsGoogleDrive import *


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


def create_beta_table(
    coin_df,
    asset_name_1,
    asset_name_2,
    calibration_window,
    frequency={"minutes": 1},
    safe_output_csv=False,
    output_folder="H:",
    n_job=4,
    stat_test="adfuller",
):
    """
    Input:
        - coinf_df : pd.DataFrame
                DataFrame with the prices of assets (coins)
        - asset_name_1 : str
                y variable
        - asset_name_2 : str
                x variable
        - calibration_window : dict
                time dictionary which determines the calibration window size
        - safe_output_csv : bool
                determines whether the output table should be saved as csv
        -stat_test : str
                string name of the statistical test to use for stationarity
    Output:
        - df_beta : pd.DataFrame
               dataframe containing beta, intercept, r2, res_std, res_mean, stationarity_pvalue, date_est
    """
    window = int(fromTimetoPlainIndex(window=calibration_window, frequency=frequency))
    coin_df = coin_df.loc[:, [asset_name_1, asset_name_2]].dropna()
    index_input = coin_df.index
    y, x = (
        coin_df[asset_name_1].to_numpy(),
        coin_df[asset_name_2].to_numpy(),
    )

    linReg = lambda x, y: linearRegression_np(x, y, stat_test=stat_test)
    rolling = npe.rolling_apply(linReg, window, x, y, n_jobs=n_job)
    df_beta = ResultDataFrame(rolling, index_input[:])
    if safe_output_csv:
        if output_folder is None or output_folder == False:
            directory = os.getcwd()
        else:
            directory = output_folder

        path_folder = os.path.join(directory, f"{asset_name_1}_{asset_name_2}")
        os.makedirs(path_folder, exist_ok=True)
        path = f"{path_folder}/df_beta_{asset_name_2}_{asset_name_1}_{window/1440}_days_{stat_test}.csv.gz"
        df_beta.to_csv(path)
        return df_beta, path
    else:
        return df_beta


def getCombRet(
    coin_df,
    asset_name_1,
    asset_name_2,
    trading_windows,
    calib_window,
    p_values,
    stop_loss=0.2,
    frequency={"minutes": 1},
    safe_beta_csv=False,
    output_folder_beta=None,
    input_folder=None,
    stat_test="adfuller",
    drive=True,
):
    """
    functions to have the result for different trading windows and threshold for a given df_beta calculated using a      calibration window
    Input:
        - coinf_df : pd.DataFrame
                DataFrame with the prices of assets (coins)
        - asset_name_1 : str
                y variable
        - asset_name_2 : str
                x variable
        - trading_windows : List
                list containing time dictionaries which determine the trading window size
        -calib_window : dict
                dict containing the calibration window to use
        - p_values : List
                list containing the relevant p-values as a threshold for trading
        - stop_loss : float in the interval of (0,1)
                determines the maximum loss we are willing to take before exiting the position
        - frequency: dict
                time dictionary with the frequency of the data
        - safe_output_csv : bool
                determines whether the output table should be saved as csv
        - output_folder_beta : str
                folder where beta table is saved if its creation is necessary
        - input_folder: str
                folder where beta tables are saved
        -stat_test : str
                statistical test to use
    Output:
        - month_dict : dict
                dictionary containing for each month the dataset of all the results and also a dictionary containing the book trades

    """
    coin_df = coin_df.loc[:, [asset_name_1, asset_name_2]].dropna()
    coin_df["Month"] = coin_df.index.to_period("M")
    index_windows = [datetime.timedelta(**window) for window in trading_windows]
    window_day = int(fromTimetoPlainIndex(window=calib_window, frequency=frequency))

    try:
        if input_folder == None or input_folder == False:
            directory = os.getcwd()
        else:
            directory = input_folder
        beta_df = pd.read_csv(
            f"{directory}/{asset_name_1}_{asset_name_2}/df_beta_{asset_name_2}_{asset_name_1}_{window_day/1440}_days_{stat_test}.csv.gz"
        )
        print("Beta Table Loaded")
        beta_df = beta_df.set_index("time")
        beta_df.index = pd.to_datetime(beta_df.index)
        beta_df["Month"] = beta_df.index.to_period("M")
    except:

        print("Calculating Beta Table")
        if drive:
            id_ = "1Vg9w6RpPjukasvRqxM4cqPxabDi9MeyS"
            folder_name, id_pair = CreatePairFolder(id_, (asset_name_1, asset_name_2))

        beta_df, path_beta = create_beta_table(
            coin_df=coin_df,
            asset_name_1=asset_name_1,
            asset_name_2=asset_name_2,
            calibration_window=calib_window,
            safe_output_csv=safe_beta_csv,
            output_folder=output_folder_beta,
            stat_test=stat_test,
        )
        if drive:
            UploadFile(file=path_beta, folder_id=id_pair)

        beta_df["Month"] = beta_df.index.to_period("M")

    df_return = pd.DataFrame(
        index=[datetime.timedelta(**w) for w in trading_windows], columns=p_values
    )
    month_dict = dict()
    for date in coin_df["Month"].unique():
        ret_dict = dict()
        df_return = pd.DataFrame(
            index=[datetime.timedelta(**w) for w in trading_windows], columns=p_values
        )
        for window in trading_windows:
            idx_window = datetime.timedelta(**window)
            for p_val in p_values:
                result_strategy = applyStrategyRolling(
                    df_beta=beta_df.loc[beta_df.Month == date],
                    df_price=coin_df.loc[coin_df.Month == date],
                    trading_window=window,
                    thr_pval=p_val,
                    quantile=2,
                    fee_rate=0.1 / 100,
                    stop_loss=stop_loss,
                )
                df_trades = strategyDataFrame(result_strategy)
                ret_dict[f"{window}_{p_val}"] = df_trades
                if len(df_trades.index) > 0:
                    total_ret = df_trades["ret_"].prod()
                else:
                    total_ret = 1
                df_return.loc[idx_window, p_val] = total_ret
        month_dict[str(date)] = (df_return, ret_dict)

    return month_dict


def borrowCost(df_, short_asset_name, enter_date, exit_date, beta=1):
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
    borrowCost_dict = joblib.load("MarginFeeCoins_Dictionary.joblib")
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
    stop_loss=0.2,
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

            borrow_fee = borrowCost(df__, asset_name_1, enter_pos_date, t_)

            fee_exit = transactionCost(
                df__, asset_name_1, asset_name_2, beta, t_, fee=fee_rate
            )

            enter_denom = np.absolute(df.loc[enter_pos_date, "spread"])
            return_no_fee = (
                -(df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"]) / enter_denom
                + 1
            )

            return_ = (
                (-borrow_fee / enter_denom)
                + (-fee_enter / enter_denom)
                + (-fee_exit / enter_denom)
                + return_no_fee
            )

            if (
                df.loc[t, "spread"] <= intercept
                or isFinaltime(j)
                or return_ <= 1 - stop_loss
            ):
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

            borrow_fee = borrowCost(df__, asset_name_2, enter_pos_date, t_, beta)

            fee_exit = transactionCost(
                df__, asset_name_1, asset_name_2, beta, t_, fee=fee_rate
            )

            enter_denom = np.absolute(df.loc[enter_pos_date, "spread"])
            return_no_fee = (
                df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"]
            ) / enter_denom + 1

            return_ = (
                (-borrow_fee / enter_denom)
                + (-fee_enter / enter_denom)
                + (-fee_exit / enter_denom)
                + return_no_fee
            )

            if (
                df.loc[t, "spread"] >= intercept
                or isFinaltime(j)
                or return_ <= 1 - stop_loss
            ):
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
    df_beta,
    df_price,
    trading_window,
    thr_pval=0.10,
    quantile=2,
    fee_rate=0.1 / 100,
    stop_loss=0.2,
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
    y_name = df_price.columns[0]
    x_name = df_price.columns[1]
    stratResult = []
    decision_trading_day = []
    # used only to initialize the for
    end_date = df_beta.index[0] - datetime.timedelta(days=1)
    delta_time = datetime.timedelta(**trading_window)
    last_start_date = df_beta.index[-2] - delta_time

    for j, t in enumerate(df_beta.index):
        row = df_beta.loc[t, :]
        if (
            row.stationarity_pvalue <= thr_pval
            and t > end_date
            and j != len(df_beta.index) - 1
        ):
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
                stop_loss=stop_loss,
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


def CreatePlot(
    result,
    asset_name_1,
    asset_name_2,
    idx,
    to_save=True,
    visualize=True,
    output_folder="H:",
    drive=True,
    size=(10, 8),
    dpi=600,
):
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = size
    plt.rcParams["figure.dpi"] = dpi

    if drive:
        try:
            files = FilesAvailableDrive(idx)
            id_ = [
                j["id"] for j in files if j["title"] == f"{asset_name_1}_{asset_name_2}"
            ][0]
            boolean_drive = True

        except:
            boolean_drive = False

    path_ = []
    cmap_reversed = cm.get_cmap("RdYlGn")

    dates = list(result.keys())
    for date in dates:
        df = result[date][0].astype("float")

        sns.heatmap(
            df,
            cmap=cmap_reversed,
            linewidths=0.5,
            annot=True,
            yticklabels=df.index,
            center=1,
            fmt=".3g",
        )
        title = f"{asset_name_1} - {asset_name_2}\n{date}"
        plt.title(title)
        plt.xlabel("P-value")
        plt.ylabel("Trading Window")

        if to_save:
            name_file = f"{asset_name_1}_{asset_name_2}_{date}.png"
            path_figure = f"{output_folder}/{asset_name_1}_{asset_name_2}/{name_file}"
            plt.savefig(path_figure)
            path_.append(path_figure)

        if visualize:
            plt.show()
        else: 
            plt.close()


    if drive and boolean_drive and to_save:
        UploadFileListData(path_, id_)
