from telnetlib import DO
from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import numpy as np
import pandas as pd
import os, sys

# outside current directory imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# StockTrading
from EnvTrade.env_trading import StockTradingEnv
# AlpacaPaperTrading
from EnvPaperTrade.env_papertrading import PaperTradingAlpaca
# DaraProcessor
from dataprocessor.data_processor import DataProcessor

ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)

state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim

API_KEY = ""
API_SECRET=""
API_BASE_URL = "https://paper-api.alpaca.markets"
data_url = 'wss://data.alpaca.markets'
env = StockTradingEnv

# pick datasource
DP = DataProcessor(
    data_source='alpaca',
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL
)

print("Datasource from alpaca: \n", DP)

# get ticker list, set start date, end date, specify data frequency
data = DP.download_data(
    start_date='2025-06-01',
    end_date='2025-06-30',
    ticker_list=ticker_list,
    time_interval='1Min'
)

print("Data contained: \n", data)

data['timestamp'].nunique()

