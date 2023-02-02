import pandas as pd
import numpy as np
from functions.LoadDataset import LoadDataset
from functools import reduce
from multiprocessing.dummy import Pool
import os


def nameFile(ticker, date, folder, frequency = "1m"):
    """
    function to get the path of the file
    Inputs:
        -ticker: str
                ticker of the dataset without USDT, e.g. "ETH"
        -date: str
                monthly date of the dataset to load, e.g. "2021-01"
        -folder: str
                path of the input folder
    Output:
        -path: str
                path of the file to load
    """
    path = f"{folder}/{ticker}USDT-{frequency}-{date}.zip"
    return path


def modifyDataFrame(x):
    """
    function to modify a dataframe and to keep only the price
    Inputs:
        -x: list or tuple
                list or tuple containing the name of the asset to load and its path
    Output:
        -df: pd.DataFrame
                dataframe containing the price of the given asset
    """
    name, path = x[0], x[1]
    df_ = LoadDataset(path)
    df_[name] = df_.close * 0.5 + df_.open * 0.5
    df = pd.DataFrame(df_[name])
    return df


def saveDataFrame(tickers, date, df, output_name=None, output_folder=None):
    """
    function to save a dataframe as .csv.gz
    the file is saved in output_folder if specified, if it is None the file is saved in the current folder
    the file is saved as 'name_date.csv.gz'
    if output_name is None then the name of the file will be the concatenation of the tickers present in the dataframe separated by an underscore, otherwise the output_name specified is used
    Inputs:
        -tickers: list
                list of the assets presents in the dataframe
        -date: str
                date of the dataframe
        -output_name: str
        -output_folder: str
    Output:
        -None

    """
    if output_folder == None:
        folder = os.getcwd()
    else:
        folder = output_folder

    if output_name == None:
        names = reduce(lambda x, y: x + "_" + y, tickers)
        name_to_save = f"{folder}/{names}_{date}.csv.gz"
        df.to_csv(name_to_save)
    else:
        name_to_save = f"{folder}/{output_name}_{date}.csv.gz"
        df.to_csv(name_to_save)

        
        
        
 def weighted_df(df, isBuyerMaker, freq):
    """
    Function to calculate the weighted average of a DataFrame.
    
    Input:
      - df: pandas DataFrame
            The DataFrame to calculate the weighted average from.
      - isBuyerMaker: bool
            Filter the DataFrame to only include rows where isBuyerMaker is equal to this value.
      - freq: str
            The frequency to resample the DataFrame to before calculating the weighted average.
            
    Output:
      - df_w: pandas DataFrame
            The weighted average of the filtered and resampled DataFrame.
    """
    df_ = df.loc[df.isBuyerMaker == isBuyerMaker]
    df_w = (df_.price * df_.qty).resample(freq).sum() / (df_.qty.resample(freq).sum())
    return df_w



def modifyDataFrameHistTrades(x, frequency):
    """
    function to modify a dataframe and to keep only the price
    Inputs:
        -x: list or tuple
                list or tuple containing the name of the asset to load and its path
    Output:
        -df: pd.DataFrame
                dataframe containing the price of the given asset
    """
    if len(frequency) == 2 and frequency[-1] == "m":
        freq = f"{frequency[0]}min"
        
    name, path = x[0], x[1]
    df = LoadDatasetHistTrades(path)
    df_to_keep = df.loc[df.isBestMatch == True]
    sell = weighted_df(df_to_keep, isBuyerMaker = True, freq = freq).rename("sell")
    buy = weighted_df(df_to_keep, isBuyerMaker = False, freq = freq).rename("buy")
    merged = pd.merge(buy, sell, left_index = True, right_index = True, how = "inner")
    df_mid = merged.mean(axis = 1).rename(name)
    return df_mid


def createDataFrame(
    input_folder,
    date,
    tickers,
    n_job=4,
    to_save=False,
    parallel=True,
    output_name=None,
    output_folder=None,
    type_ = "klines", 
    frequency = "1m"
):
    """
    function to create and save the dataframe
    Inputs:
        -input_folder: str
                folder from which retrieve the data
        -date: str
                date for which create the dataframe
        -tickers: list of str
                list of assets name, e.g. ["ETH", "BTC"]
        -n_job: int
                cores used to parallelize the operations
        -to_save: bool
                boolean variable describing if the dataset is to save
        -output_name: str
                custom output name, e.g. "testdataset", note this is the name not the path, if none an automatic name is used, check the function saveDataFrame
        -output_folder: str
                path of the folder where to save the data, if none the current folder is used
    Output:
        df: pd.DataFrame
                pandas dataframe having the time as index and as many columns and the number of tickers specified in input

    """
    if type_ == "klines":
        paths = [nameFile(ticker, date, input_folder, frequency = frequency) for ticker in tickers]
        fun = lambda x: modifyDataFrame(x)
    elif type_ == "trades":
        paths = [nameFileHistTrades(ticker, date, input_folder) for ticker in tickers]
        fun = lambda x: modifyDataFrameHistTrades(x, frequency = frequency)
        
    if parallel:
        dfs = Pool(n_job).map(fun, zip(tickers, paths))
    else:
        dfs = [fun(x) for x in zip(tickers, paths)]

    df = reduce(
        lambda x, h: pd.merge(x, h, right_index=True, left_index=True, how="outer"), dfs
    )
    if to_save:
        saveDataFrame(tickers, date, df, output_name, output_folder)
    return df


def createUniqueDataFrame(list_dates, input_folder, tickers, n_job=4, to_save=False, 
                          output_name=None, output_folder=None, type_ = "klines", frequency = "1m"):
    """
    Function to concatenate and merge multiple dataframes of multiple tickers
    into a single unique dataframe. 
    Inputs:
        -list_dates: list of str
                list of dates to consider for the dataframes, e.g. ["2021-01", "2021-02"]
        -input_folder: str
                path of the input folder where the dataframes are stored
        -tickers: list of str
                list of tickers to consider for the dataframes, e.g. ["BTC", "ETH"]
        -n_job: int
                number of jobs to parallelize the process
        -to_save: bool
                if True the dataframe will be saved
        -output_name: str
                name of the output file to save, if None will use the default naming convention
        -output_folder: str
                path of the output folder where the dataframe will be saved, if None will use the input_folder
        -type_: str
                type of data to be considered, e.g. "klines"
        -frequency: str
                frequency of the data to be considered, e.g. "1m"
    Output:
        -unique_df: pandas dataframe
                unique dataframe of the input dates and tickers
    """
    fun_ = lambda date: createDataFrame(input_folder=input_folder, date=date, tickers=tickers, n_job=n_job, 
                                        to_save=False, type_ = type_, frequency = frequency)
    unique_df = pd.concat([fun_(date) for date in list_dates])
    if to_save:
        first_last_date = list_dates[0] + "_" + list_dates[-1]
        saveDataFrame(tickers, first_last_date, unique_df, output_name, output_folder)
    return unique_df


def LoadDatasetHistTrades(path):
    """
    Function to load a dataset and to set the time as index
    names of the columns are taken from: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#old-trade-lookup-market_data
    Input:
      -path: str
            path of the zipped dataset to open
    Output: 
      -df : pd.DataFrame
    
    """
    df = pd.read_csv(path, header = None)
    df.columns = [ "Id"	,"price",	"qty",	"quoteQty",	"time"	,"isBuyerMaker",	"isBestMatch"]
    df["time"] = pd.to_datetime(df["time"].to_numpy(), utc = True, unit = "ms")
    df.index = df["time"]
    df.drop(columns = ["time"], inplace = True)
    return df 



def nameFileHistTrades(ticker, date, folder):
    """
    function to get the path of the file
    Inputs:
        -ticker: str
                ticker of the dataset without USDT, e.g. "ETH"
        -date: str
                monthly date of the dataset to load, e.g. "2021-01"
        -folder: str
                path of the input folder
    Output:
        -path: str
                path of the file to load
    """
    path = f"{folder}/{ticker}USDT-trades-{date}.zip"
    return path



def loadCleanDataFrame(path):
    """
    Load and clean a CSV dataframe from the specified file path.

    Inputs:
        - path: str
                The file path of the CSV file.

    Output:
        - df_: pandas.DataFrame
                The loaded and cleaned dataframe.
    """
    df_ = pd.read_csv(path)
    df_ = df_.set_index(pd.to_datetime(df_.time))
    df_.drop(columns=["time"], inplace=True)
    return df_

