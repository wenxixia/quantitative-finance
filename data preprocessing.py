#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:59:02 2021

@author: wenxixia
"""
from io import StringIO
import os
import pandas as pd
import numpy as np
import requests
import pickle
import urllib.request
import pandas as pd
import numpy as np


dir = '/Users/wenxixia/Desktop/capstone project/'

def fetch_data(symbol):

    time_frame = "d" 
    # d for daily data, w for weekly, m for monthly.
    url = "https://query1.finance.yahoo.com/v7/finance/download/"+symbol+\
    "?period1=1262352723&period2=1635043123&interval=1d&events=history&includeAdjustedClose=true"

    response = requests.get(url,headers={'User-agent': 'Mozilla/5.0'})
    pd.read_csv(StringIO(response.text),index_col = "Date",parse_dates = ["Date"]).to_csv(dir+'raw_data/'+symbol+'.csv')
    
    print(response.text[:100])
    print("Downloading for "+ symbol)

def DownloadPrices():
    stocklist = pd.read_csv(dir+'S&P500 stocklist.csv')
 
    for symbol in stocklist["Symbol"]:
        try:
            fetch_data(symbol)
        except:
            print(symbol)

def fill_missing_values(df_data):
    df_data.fillna(method="ffill",inplace=True)
    df_data.fillna(method="bfill", inplace=True)


def symbol_to_path(symbol, base_dir=dir+"raw_data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        print("Fetching {}".format(symbol))
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def get_rolling_mean(values, window,min_periods=None):
    return values.rolling(window=window, min_periods=min_periods).mean()


def get_rolling_std(values, window,min_periods=None):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window, min_periods=min_periods).std()


def get_bollinger_bands(rm, window,min_periods=None):
    rstd=get_rolling_std(rm, window, min_periods)
    upper_band=rm+rstd*2
    lower_band = rm - rstd * 2
    return upper_band, lower_band

def getAllStocks():
    stocklist = pd.read_csv(dir+'S&P500 stocklist.csv')
    dates = pd.date_range('2016-01-01', '2021-10-21')
    all = get_data(stocklist["Symbol"].values.tolist(),dates)
    print(all.head())
    all.to_pickle(dir+'data/all_unprocessed.pkl')
    print("Saved!")

def getPrices():
    stocklist = pd.read_csv(dir+'S&P500 stocklist.csv')
    dates = pd.date_range('2016-01-01', '2021-10-21')
    all = get_data(stocklist["Symbol"].values.tolist(),dates)
    fill_missing_values(all)
    #all.to_pickle(dir+'data/prices.pkl')
    #print("{} Prices Saved!".format(len(all)))
    return all

def CalculateAdjSMAs(df):
    for w in [10, 20, 50, 200]:
        sma = get_rolling_mean(df, w, 0)
        adjsma = df / sma-1
        adjsma.to_pickle(dir+'data/adjsma{}.pkl'.format(w))
        print("AdjSMA {} Saved!".format(w))


def CalculateBollingerBands(df):
    for w in [10, 20, 50, 200]:
        bbu,bbl = get_bollinger_bands(df, w,0)
        bbu.fillna(method="bfill", inplace=True)
        bbl.fillna(method="bfill", inplace=True)
        bbub=bbu<df
        bblb = bbl > df
        bbub=bbub.applymap(lambda x: 1.0 if x else 0.0)
        bblb=bblb.applymap(lambda x: 1.0 if x else 0.0)
        bb=bbub-bblb
        bb.to_pickle(dir+'data/bb{}.pkl'.format(w))

        print("Bollinger Bands {} Saved!".format(w))


def CalculateStates(df,state_size_day,datanames,date_test,save=True):
    date_test_conv=np.datetime64(date_test+'T21:00:00.000000000-0300')
    data = []
    train_states = pd.DataFrame(columns=["State"])
    test_states = pd.DataFrame(columns=["State"])
    nfeatures=0

    print("Calculating States!")

    for name in datanames:
        with open(dir+'data/{}.pkl'.format(name), "rb") as fh:
            x = pickle.load(fh)
            data.append(x)
            
    for i in range(state_size_day-1,len(df)):
        rest_of_features=[]
        for d in data:
            for j in range(i-state_size_day+1,i+1):
                rest_of_features.extend(d.iloc[j].values)
                print(d.iloc[j])
        state=[df.index.values[i]]
        state.extend(rest_of_features)
        print(rest_of_features)
        print(state)
        nfeatures = len(state)
        if state[0]<=date_test_conv:
            train_states.loc[len(train_states)] = [state]
        else:
            test_states.loc[len(test_states)] = [state]
    if save:
        print("Saving Train Data")
        #train_states.to_pickle(dir+"data/train.pkl")
        print("Saving Test Data")
        #test_states.to_pickle(dir+"data/test.pkl")
        print("States Saved! {} dimensions".format(nfeatures))
    
    return train_states,test_states



if __name__ == "__main__":
    DownloadPrices()
    #df=getPrices()
    #CalculateAdjSMAs(df)
   # CalculateBollingerBands(df)
    #CalculateStates(df,1,['adjsma20','bb20'],'2021-01-01')
    #getAllData(7,['adjsma20','bb20'])
    #print(getRandomHoldCombinations(10))