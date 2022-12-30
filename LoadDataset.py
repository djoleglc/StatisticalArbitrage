import pandas as pd 
import numpy as np 


def LoadDataset(path):
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
    df.columns = ["id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"]
    df["time"] = pd.to_datetime(df["time"].to_numpy(), utc = True, unit = "ms")
    df.index = df["time"]
    df.drop(columns = ["time"], inplace = True)
    return df 
