import json
import joblib
import warnings
from multiprocessing.dummy import Pool
import os
from itertools import combinations
import pandas as pd
from functions.UtilsRetrieveData import downloadList, create_listdates
from functions.UtilsGoogleDrive import *
from functions.UtilsCreateDataFrame import createUniqueDataFrame
from functions.Strategy import *
import matplotlib

def open_config(path):
    with open(path) as json_file:
        to_return = json.load(json_file)
    return to_return


def load_assets(Mconfig):
    assets = joblib.load(Mconfig["list_assets_path"])
    if Mconfig["toy"]:
        return assets[0:3]
    else:
        return assets


def retrieve_data(assets, Mconfig, dates):
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
        type_="klines",
        frequency=Mconfig["frequency"],
    )
    return df


def create_result_folder(output_folder):
    path = f"{output_folder}/results"
    os.makedirs(path, exist_ok=True)
    return path


def create_pairs(assets_list):
    pairs = [comb for comb in combinations(assets_list, r=2)]
    return pairs


def apply_strategy(coin_df, pair, Mconfig, config, path_result):

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
    # applying the strategy
    result = apply_strategy(
        coin_df=coin_df,
        pair=pair,
        Mconfig=Mconfig,
        config=config,
        path_result=path_result,
    )

    # create the plots
    CreatePlot(
        result,
        asset_name_1=pair[0],
        asset_name_2=pair[1],
        to_save=True,
        visualize=False,
        output_folder=path_result,
        drive=False,
        idx=None,
        dpi=80,
    )
    
    
def beta_table(pair, coin_df, Mconfig, config, path_result):
        
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

        if Mconfig["drive"]:
            UploadFile(file=path_beta, folder_id=id_pair)
    

    
def retrieve_clean_dataset(Mconfig):
    #retrieve data from google drive 
    idx = Mconfig[f"idx_{Mconfig['type_data']}"]
    files =  FilesAvailableDrive(idx)
    RetrieveFile(files, idx)
    #create the clean folder
    clean_folder = f"{Mconfig['output_folder']}/cleandata"
    os.makedirs(clean_folder, exist_ok=True)
    #move the file 
    shutil.move(files[0]["title"], clean_folder)
    #load the file 
    df = loadCleanDataFrame(f"{clean_folder}/{files[0]['title']}")
    return df 
                
    
    
    

def main():
    matplotlib.use('agg')
    # load configuration run main
    Mconfig = open_config(path="config/Mconfig.json")
    config = open_config(path="config/config.json")
    
    # load assets to use
    assets = load_assets(Mconfig)
    # creating list of pairs
    pairs = create_pairs(assets)
    # create list of dates
    dates = create_listdates(Mconfig["dates_interval"])
    
    if Mconfig["toy"]:
        # if toy example run consider only three months
        if len(dates) >= 3:
            dates = dates[:3]
            
            
    # retrieve the data
    if Mconfig["download_clean_data"] == False:
        print("Downloading raw data")
        retrieve_data(assets, Mconfig, dates)
        # create clean dataframe
        coin_df = create_unique_dataset(Mconfig, assets, dates)
    else:
        print("Loading clean data")
        coin_df = retrieve_clean_dataset(Mconfig)
    
    # create result folder
    path_result = create_result_folder(Mconfig["output_folder"])
    # calculate the beta table

    function = lambda pair: apply_strategy_pair(
        pair, coin_df, Mconfig, config, path_result
    )
    
    #calculate the beta for each pair
    for pair in pairs:
        print("\nCalculating Beta table for: ", pair)
        beta_table(pair, coin_df, Mconfig, config, path_result)

    #applying the strategy in parallel
    print("Applying the Strategy")
    Pool(Mconfig["n_job"]).map(function, pairs)


main()
