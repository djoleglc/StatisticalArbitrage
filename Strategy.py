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
        
        
def transactionCost(df_, asset_name_1, asset_name_2, beta, t, fee = 0.1/100):
    """
    asset 1 is the y
    asset 2 is the x 
    """
    total_amount = df_.loc[t, asset_name_1] + beta * df_.loc[t, asset_name_2]
    return total_amount * fee


def apply_twosigma(
    df_, asset_name_1, asset_name_2, beta, intercept, sigma, start_date, end_date, fee_rate = 0.1/100
):
    """
    index of df must be datetime
    version with multiple trades in a trading period
    asset_one is the y 
    asset_two is the x
    """
    if end_date <= df_.index[-1]:
        df__ = df_.loc[start_date:end_date, :]
    else: 
        df__ = df.loc[start_date:, : ]
    
    df = pd.DataFrame(
        df__.loc[:, asset_name_1] - beta * df__.loc[:, asset_name_2] ,
        columns=["spread"],
        index=df__.index,
    )
        
    state = 0  # initial state
    result = ResultStrategy([], [], [], [])

    for j,t in enumerate(df.index[:-1]):
        if state == 0:
            if df.loc[t, "spread"] > intercept + (2 * sigma):
                # here we SHORT
                state = -1
                enter_pos_date = df.index[j+1]
                result.enter_.append(enter_pos_date)

            if df.loc[t, "spread"] < intercept + (-2 * sigma):
                # here we go LONG
                state = 1
                enter_pos_date = df.index[j+1]
                result.enter_.append(enter_pos_date)

        elif (state == -1 and df.loc[t, "spread"] <= intercept) or (
            state == -1 and j == len(df) - 2
        ):
            state = 0
            t_ = df.index[j+1]
#             print("ENTER: ", df.loc[enter_pos_date, "spread"])
#             print("EXIT:  ", df.loc[t_, "spread"])
#             print("\n")
            fee_cost = transactionCost(df__, asset_name_1, asset_name_2, beta, t_, fee = fee_rate)
            return_ = (-fee_cost/ df.loc[enter_pos_date, "spread"]) - (df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"] ) / df.loc[enter_pos_date, "spread"] + 1
            result.ret_.append(return_)
            result.ret_type.append("Short")
            result.exit_.append(t_)

        elif (state == 1 and df.loc[t, "spread"] >= intercept) or (
            state == 1 and  j == len(df) - 2
        ):
            state = 0
            t_ = df.index[j+1]
#             print("ENTER: ", df.loc[enter_pos_date, "spread"])
#             print("EXIT:  ", df.loc[t_, "spread"])
#             print("\n")
            fee_cost = transactionCost(df__, asset_name_1, asset_name_2, beta, t_, fee = fee_rate)
            return_ = (-fee_cost / df.loc[enter_pos_date, "spread"]) + ( df.loc[t_, "spread"] - df.loc[enter_pos_date, "spread"] ) / df.loc[enter_pos_date, "spread"] + 1
            result.ret_.append(return_)
            result.exit_.append(t_)
            result.ret_type.append("Long")

    return result



def applyStrategyRolling(df_beta,df_price, trading_window_day = 1, frequency = "1m", thr_pval = 0.10):
    
    y_name = df_price.columns[1]
    x_name = df_price.columns[0]
    stratResult = []
    decision_trading_day = []
    #used only to initialize the for
    end_date = df_beta.index[0] - datetime.timedelta(days=1)
    last_start_date = df_beta.index[-2] - datetime.timedelta(days = trading_window_day)
    
    for j,t in enumerate(df_beta.index):
       # if t <= last_start_date:
            row = df_beta.loc[t, :]
            if row.stationarity_pvalue <= thr_pval and  t > end_date: 
                decision_trading_day.append(row)
                init_date = df_beta.index[j+1]
                end_date = init_date +  datetime.timedelta(days=trading_window_day)

                strat =  apply_twosigma(
                        df_price,
                        y_name,
                        x_name,
                        beta=row.beta,
                        intercept=row.intercept,
                        sigma = row.res_std,
                        start_date= init_date,
                        end_date = end_date,
                    )
                stratResult.append(strat)


    return stratResult, decision_trading_day
