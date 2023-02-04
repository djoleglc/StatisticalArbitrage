import json
import joblib
import warnings
from multiprocessing.dummy import Pool
import os
import itertools
from itertools import combinations
import pandas as pd
from functions.UtilsRetrieveData import downloadList, create_listdates
from functions.UtilsGoogleDrive import *
from functions.UtilsCreateDataFrame import createUniqueDataFrame
from functions.Strategy import *
import matplotlib
import matplotlib.dates as mdates
import random
import datetime as dt


def open_config(path):
    """
    Loads a JSON configuration file from the given path and returns its contents as a Python object.

    Input:
      - path: str
        The path to the JSON configuration file to be opened.

    Output:
      - to_return: object
        The contents of the JSON configuration file as a Python object.
    """
    with open(path) as json_file:
        to_return = json.load(json_file)
    return to_return


def retrieve_data(assets, Mconfig, dates):
    """
    Downloads data for a list of assets for a specified set of dates, and saves the data in a specified folder.

    Input:
      - assets: list or array-like
        A list or array-like object containing the assets for which data should be retrieved.
      - Mconfig: dict
        A dictionary that contains the configuration parameters, including the number of parallel jobs to run, the type of data to retrieve, and the frequency of data to retrieve.
      - dates: list or array-like
        A list or array-like object containing the dates for which data should be retrieved.

    Output:
      None
    """
    n_job = Mconfig["n_job"]
    raw_folder = f"{Mconfig['output_folder']}/rawdata"
    os.makedirs(raw_folder, exist_ok=True)
    for name in assets:
        print(f"Retrieving {name}")
        paths = downloadList(
            name=f"{name}USDT",
            dates=dates,
            n_job=n_job,
            type_=Mconfig["type_data"],
            folder=raw_folder,
            frequency=Mconfig["frequency"],
            return_path=True,
        )
        print("\n")


def create_unique_dataset(Mconfig, assets, dates):
    """
    Function to create a unique dataset of all assets and to save it.
    Input:
      - Mconfig: dict
            contains all the configs of the experiment
      - assets: list
            list of the assets to consider
      - dates: list
            list of the dates to consider
    Output:
      - df : pd.DataFrame
            dataframe containing all the data of the assets
    """
    input_folder = f"{Mconfig['output_folder']}/rawdata"
    clean_folder = f"{Mconfig['output_folder']}/cleandata"
    os.makedirs(clean_folder, exist_ok=True)
    to_save = Mconfig["save_dataframe"]
    n_job = Mconfig["n_job"]
    tickers = assets

    df = createUniqueDataFrame(
        list_dates=dates,
        input_folder=input_folder,
        tickers=tickers,
        n_job=n_job,
        to_save=to_save,
        output_name=f"AllTickers_klines_{Mconfig['frequency']}",
        output_folder=clean_folder,
        type_=Mconfig["type_data"],
        frequency=Mconfig["frequency"],
    )
    return df


def create_result_folder(output_folder):
    """
    Function to create a folder to store the results.

    Input:
      - output_folder: str
            Path to the main output folder.

    Output:
      - path: str
            Path to the folder for storing results.
    """
    path = f"{output_folder}/results"
    os.makedirs(path, exist_ok=True)
    return path


def create_pairs(Mconfig):
    """
    Function to load and return a list of pairs from the stored file.

    Input:
      - assets_list: list
            List of assets.

    Output:
      - pairs: list
            List of pairs.
    """
    pairs = joblib.load("config/all_pairs.joblib")
    if Mconfig["toy"]:
        # let's take randomly three pairs
        pairs = random.sample(pairs, 3)
    return pairs


def apply_strategy(coin_df, pair, Mconfig, config, path_result):
    """
    Function to apply a trading strategy and save the results.

    Input:
      - coin_df: pd.DataFrame
            DataFrame containing the data for a particular asset.
      - pair: tuple
            Tuple of two asset names as a pair.
      - Mconfig: dict
            Dictionary containing the main configuration parameters.
      - config: dict
            Dictionary containing the configuration parameters for the strategy.
      - path_result: str
            Path to the folder for storing results.

    Output:
      - result: dict
            Dictionary containing the results of the strategy.
    """
    result = getCombRet(
        coin_df=coin_df,
        asset_name_1=pair[0],
        asset_name_2=pair[1],
        trading_windows=config["tresh"],
        calib_window=config["calibration_window"],
        p_values=config["p_values"],
        stop_loss=config["stop_loss"],
        stat_test=config["stat_test"],
        safe_beta_csv=True,
        input_folder=path_result,
        output_folder_beta=path_result,
        verbose=True,
        drive=False,
    )

    SaveUploadResultStrategy(
        result, pair[0], pair[1], drive=False, output_folder=path_result, idx=None
    )

    return result


def apply_strategy_pair(pair, coin_df, Mconfig, config, path_result):
    """
    Function to apply the strategy and create the plots for a given pair.

    Input:
      - pair: list
            A list of two elements representing a single trading pair.
      - coin_df: pandas.DataFrame
            DataFrame containing the asset data.
      - Mconfig: dict
            Dictionary containing the global configurations.
      - config: dict
            Dictionary containing the configurations specific to the strategy.
      - path_result: str
            Path to the folder where results are stored.

    Output:
      - None
    """
    # applying the strategy
    result = apply_strategy(
        coin_df=coin_df,
        pair=pair,
        Mconfig=Mconfig,
        config=config,
        path_result=path_result,
    )


def beta_table(pair, coin_df, Mconfig, config, path_result):
    """
    Function to create a beta table and save it to the specified path.

    Input:
      - pair: tuple
            Tuple of two assets to form a pair.
      - coin_df: pandas dataframe
            Dataframe containing price information for the assets.
      - Mconfig: dict
            Dictionary of main configuration parameters.
      - config: dict
            Dictionary of configuration parameters specific to this function.
      - path_result: str
            Path to store the results of the function.

    Output:
      - beta_table: pandas dataframe
            Dataframe containing beta information for the pair.
      - path_beta: str
            Path to the saved beta table.
    """
    beta_table, path_beta = create_beta_table(
        coin_df=coin_df,
        asset_name_1=pair[0],
        asset_name_2=pair[1],
        calibration_window=config["calibration_window"],
        frequency=config["frequency"],
        safe_output_csv=True,
        n_job=Mconfig["n_job"],
        output_folder=path_result,
        stat_test=config["stat_test"],
    )

    return beta_table, path_beta


def retrieve_clean_dataset(Mconfig, config):
    """
    Function to retrieve and return the clean dataset from a google drive file.

    Input:
      - Mconfig: dict
            Dictionary of configuration parameters.
      - config: dict
            Dictionary of configuration parameters.

    Output:
      - df: pandas DataFrame
            Clean dataset.
    """
    # retrieve data from google drive
    idx = config[f"idx_{Mconfig['type_data']}"]
    files = FilesAvailableDrive(idx)
    RetrieveFile(files, idx)
    # create the clean folder
    clean_folder = f"{Mconfig['output_folder']}/cleandata"
    os.makedirs(clean_folder, exist_ok=True)
    # move the file
    if files[0]["title"] not in os.listdir(clean_folder):
        shutil.move(files[0]["title"], clean_folder)
    else:
        os.remove(f"{clean_folder}/{files[0]['title']}")
        shutil.move(files[0]["title"], clean_folder)
    # load the file
    df = loadCleanDataFrame(f"{clean_folder}/{files[0]['title']}")
    return df


def load_result(asset_1, asset_2, out_fold):
    """
    Function to load the result from a joblib file.

    Input:
      - asset_1: str
            Name of the first asset in the pair.
      - asset_2: str
            Name of the second asset in the pair.
      - out_fold: str
            Path to the folder where the results are stored.

    Output:
      - result: object
            Result loaded from the joblib file.
    """
    result_path = f"{out_fold}/{asset_1}_{asset_2}/{asset_1}_{asset_2}.joblib"
    result = joblib.load(result_path)
    return result


def plot_example(n, frequency, asset_1, asset_2, result, coin_df, outfolder, save=True):
    """
    Function to plot the example of a trading strategy.

    Input:
    - n: int
            Number of trades to plot.
    - frequency: str
            Frequency of the trades.
    - asset_1: str
            Name of the first asset in the pair.
    - asset_2: str
            Name of the second asset in the pair.
    - result: object
            Result of the backtest of the trading strategy.
    - coin_df: pandas DataFrame
            Data of the coins.
    - outfolder: str
            Path to the folder where the plot will be saved.
    - save: bool (optional)
            Save the plot or not. Default is True.

    Output:
    None
    """
    r = result["2021-01"][1]
    freq = {"hours": 12}
    key = f"{freq}_0.05"
    trades = r[key]
    xmft = mdates.DateFormatter("%H:%M")

    c = 0
    for j in range(len(trades.beta.unique()[:n])):
        c += 1
        beta = trades.beta.unique()[j]
        # print(trades.loc[trades.beta == beta].enter_)

        intercept = trades.intercept.unique()[j]
        sigma = trades.res_std.unique()[j]

        r2 = trades.r2.unique()[j]
        tr = trades.loc[trades.beta == beta]
        if c == 2:
            tr_check = tr
        ret = np.round(trades.loc[trades.beta == beta].ret_.prod(), 4)
        starting = pd.to_datetime(tr.date_est.iloc[0]) + frequency

        ending = starting + dt.timedelta(**freq)
        list_open_close = [
            (open_, close_) for open_, close_ in zip(tr.enter_, tr.exit_)
        ]
        full_list = []

        if starting + dt.timedelta(**{"minutes": 30}) > list_open_close[0][0]:
            starting = starting - dt.timedelta(**{"minutes": 30})
        df = coin_df.loc[starting:ending, [asset_1, asset_2]]
        ind = df.index.to_pydatetime()

        for open_, close_ in list_open_close:
            full = pd.date_range(start=open_, end=close_, freq="1min")
            full_list.append(full)

        estim = df[asset_2] * beta + intercept
        plt.plot(ind, df[asset_1], linewidth=1.5, color="blue", label=asset_1)
        alpha_beta = r"$\hat{\alpha} + \hat{\beta}$"
        plt.plot(
            ind, estim, linewidth=1.5, color="red", label=f"{alpha_beta} {asset_2}"
        )
        y_lim = (
            min(estim.min(), df[asset_1].min()) * 0.99,
            max(estim.max(), df[asset_1].max()) * 1.02,
        )
        plt.ylim = y_lim
        plt.fill_between(
            ind,
            df[asset_1] + 2 * sigma,
            y_lim[1],
            color="lightgrey",
            label=r"Outside 2$\hat{\sigma}$ interval",
        )
        plt.fill_between(ind, df[asset_1] - 2 * sigma, y_lim[0], color="lightgrey")
        plt.grid(False)
        plt.margins(y=0)
        plt.margins(x=0)

        for idx, full in enumerate(full_list):
            if tr.ret_type.iloc[idx] == "Long":
                color = "lightgreen"
            else:
                color = "skyblue"
            try:
                plt.fill_between(
                    full,
                    df.loc[full, asset_1],
                    df.loc[full, asset_2] * beta + intercept,
                    color=color,
                )
            except:
                pass
        fontsize = 16
        plt.title(
            f"{asset_1} - {asset_2} \nTotal Return Trading Window: {ret} \nR$^2$: {np.round(r2,3)}",
            fontsize=fontsize,
        )
        plt.xlabel("Time", fontsize=fontsize)
        plt.ylabel("Price", fontsize=fontsize)
        plt.gca().xaxis.set_major_formatter(xmft)
        plt.rcParams["xtick.labelsize"] = fontsize
        plt.rcParams["ytick.labelsize"] = fontsize
        plt.legend(
            bbox_to_anchor=(1.025, 0.7), loc="upper left", prop={"size": fontsize}
        )
        plt.savefig(
            f"{outfolder}/{asset_1}_{asset_2}/{asset_1}_{asset_2}_trade_plot_{c}.png",
            bbox_inches="tight",
        )
        plt.close()


def main():
    # Use the "agg" backend for Matplotlib
    matplotlib.use("agg")

    # Load configuration files
    Mconfig = open_config(path="config/Mconfig.json")
    config = open_config(path="config/config.json")

    # Set default output folder if not specified in Mconfig
    if Mconfig["output_folder"] is None:
        Mconfig["output_folder"] = os.getcwd()

    # Create list of pairs from Mconfig
    pairs = create_pairs(Mconfig)

    # Get unique assets from pairs
    assets = list(set(itertools.chain(*pairs)))

    # Create list of dates from config
    dates = create_listdates(config["dates_interval"])

    # Truncate dates list to first three if in "toy" mode
    if Mconfig["toy"]:
        if len(dates) >= 3:
            dates_ = dates
            dates = dates[:3]
            next_date = dates_[3]

    # Retrieve or load data depending on Mconfig
    if Mconfig["download_clean_data"] == False:
        print("Downloading raw data")
        retrieve_data(assets, Mconfig, dates)
        coin_df = create_unique_dataset(Mconfig, assets, dates)
    else:
        print("Loading clean data")
        coin_df = retrieve_clean_dataset(Mconfig, config)
        if Mconfig["toy"]:
            coin_df = coin_df[coin_df.index < next_date]

    # Create folder for results
    path_result = create_result_folder(Mconfig["output_folder"])

    # Define function for applying strategy in parallel
    function = lambda pair: apply_strategy_pair(
        pair, coin_df, Mconfig, config, path_result
    )

    # Calculate beta table for each pair
    for pair in pairs:
        print("\nCalculating Beta table for: ", pair)
        beta_table(pair, coin_df, Mconfig, config, path_result)

    # Apply strategy in parallel
    print("Applying the Strategy")
    Pool(Mconfig["n_job"]).map(function, pairs)

    frequency = {"minutes": 1}
    frequency = dt.timedelta(**frequency)

    # Create plots for each pair
    for pair in pairs:
        
        asset_1 = pair[0]
        asset_2 = pair[1]
        res = load_result(asset_1, asset_2, path_result)

        # Create plot with `CreatePlot` function
        CreatePlot(
            res,
            asset_name_1=asset_1,
            asset_name_2=asset_2,
            to_save=True,
            visualize=False,
            output_folder=path_result,
            drive=False,
            idx=None,
            dpi=80,
        )

    plt.rcParams["figure.figsize"] = (20, 7)
    for pair in pairs:
        # create example of plots
        asset_1 = pair[0]
        asset_2 = pair[1]
        res = load_result(asset_1, asset_2, path_result)

        
        plot_example(
            5, frequency, asset_1, asset_2, res, coin_df, path_result, save=True
        )


if __name__ == "__main__":
    main()
