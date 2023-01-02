import pandas as pd
import numpy as np
from LoadDataset import LoadDataset
from functools import reduce
from multiprocessing.dummy import Pool
import os


def nameFile(ticker, date, folder):
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


def modifyDataFrame(x, frequency):
    """
    function to modify a dataframe and to keep only the price
    Inputs:
        -x: list or tuple
                list or tuple containing the name of the asset to load and its path
        -frequency: str
                pandas frequency to use to resample the dataset
    Output:
        -df: pd.DataFrame
                dataframe containing the price of the given asset
    """
    name, path = x[0], x[1]
    df = pd.DataFrame(LoadDataset(path).price)
    df = df.rename(columns={"price": name})
    df = df.resample(frequency).last()
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


def createDataFrame(
    input_folder,
    date,
    tickers,
    frequency,
    n_job=4,
    to_save=False,
    output_name=None,
    output_folder=None,
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
        -frequency: str
                pandas frequency used to resample the data
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
    paths = [nameFile(ticker, date, input_folder) for ticker in tickers]
    fun = lambda x: modifyDataFrame(x, frequency)
    dfs = Pool(n_job).map(fun, zip(tickers, paths))
    df = reduce(
        lambda x, h: pd.merge(x, h, right_index=True, left_index=True, how="outer"), dfs
    )
    if to_save:
        saveDataFrame(tickers, date, df, output_name, output_folder)
    return df
