import time
import requests
import os
from tqdm import tqdm
import requests
import shutil
import wget
from multiprocessing.dummy import Pool


def create_listdates(d):
    """
    Function to create list of dates given first and last dates to takes
    Inputs:
        -d: list
                list containing the first and the last date to take, single date must have syntax e.g. "2021-02"
    Output:
        -dates: list
                list containing all the dates to retrieve
    """
    first_split = d[0].split("-")
    last_split = d[1].split("-")
    fy, fm = int(first_split[0]), int(first_split[1])
    ly, lm = int(last_split[0]), int(last_split[1])

    dates = []
    dates.append(d[0])
    month = fm
    year = fy

    while True:
        month += 1
        if month > 12:
            month = 1
            year += 1

        dates.append(f"{year}-{str(month).zfill(2)}")
        if year == ly and month == lm:
            break
    return dates


def downloadFile(name, date):
    """
    Function to download a list of dates files using Parallelization
    filename is setted for this project
    Inputs:
        -name: str
                name of the file to download, e.g. ETHUSDT
        -date: str
                date to download date must have the following syntax: "2021-01"
    """
    directory = os.getcwd() 
    filename = f"{directory}\\{name}-1m-{date}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{name}/1m/{name}-1m-{date}.zip"
    wget.download(url)
    return filename
    

    
def moveFile(filename, name, date, folder):
    """
    Function to move a file 
    Inputs:
        -filename: str
                path of the file to move 
        -name: str
                name of the currency of the file to move
        -date: str
                date of the file to move 
        -folder: str
                name of the folder to store the data, e.g. 'H'
           
    """
    shutil.move(filename, f"{folder}\\{name}-1m-{date}.zip")

    
def downloadmoveFile(name, date, folder = "H:"):
    """
    Function to download and move a file
    Inputs:
        -name: str
                name of the currency to retrieve
        -date: str
                date to retrieve, e.g. '2021-05'
        -folder: str
                folder to store the file
    """
    filename = downloadFile(name, date)
    moveFile(filename, name, date, folder)
    
    
    

def downloadList(name, dates, parallel=True, n_job=2):
    """
    Function to download a list of dates files using Parallelization
    Inputs:
        -name: str
                name of the file to download, e.g. ETHUSDT
        -dates: list
                list with the dates to download, single date must have the following syntax: "2021-01"
        -parallel: bool
                boolean variable that describe if parallelization must be used
        -n_job: int
                number of cores to use
    """
    fun_ = lambda date: downloadmoveFile(name, date)
    if parallel:
        Pool(n_job).map(fun_, dates)
        return "Completed"
    else:
        for date in dates:
            fun_(date)
        return "Completed"
