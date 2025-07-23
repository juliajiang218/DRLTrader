"""
How to run this file:

python3 -m venv ~/yfinance
source ~/yfinance/bin/activate
pip3 install yfinance


"""

from main import preprocess_data, create_env_kwargs
from stable_baselines3 import A2C, DDPG, PPO
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
import sys, os
import pandas as pd
from utils import *
import matplotlib.pyplot as plt

# allows to import modules/packages from project root and its subdirectories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # without this, python can't find modules located outside main.py directory
from preprocessor.yahoodownloader import YahooDownloader
from env_stock_trading.env_stocktrading import StockTradingEnv
from agents.DRLAgent import DRLAgent

# load test dataset
_, train, test = preprocess_data()

# load trained models
trained_a2c = A2C.load("/deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/trained_models/AGENT_a2c.zip")
trained_ppo = PPO.load("/deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/trained_models/AGENT_ppo.zip")

# use trained models to trade on test dataset
    # create trading environment, based on test dataset
env_kwargs, stock_dimension, state_space = create_env_kwargs(test)
test_env = StockTradingEnv(df=test,turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
env_trade, _ = test_env.get_sb_env()

    # start testing
df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c,
    environment= test_env
)

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo,
    environment= test_env
)

# ---------use MVO, DIJA to trade on test dataset
# ----MVO:
#calculate weights for mean-variance
stockData = process_mvo_df(train, stock_dimension)
tradeData = process_mvo_df(test, stock_dimension)

tradeData.to_numpy()

print(tradeData)

#compute asset returns
import numpy as np

mvoStockPrices = np.asarray(stockData)
[rows, cols] = mvoStockPrices.shape
mvoReturns = StockReturnsComputing(mvoStockPrices, rows, cols)

# compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(mvoReturns, axis=0)
covReturns = np.cov(mvoReturns, rowvar=False)

#set precision for printing results
np.set_printoptions(precision=3, suppress=True)

#display mean returns, variance-covariance matrix of returns
print("Mean returns of assets in k-portfolio \n", meanReturns)
print("Variance-covariance matrix of returns: \n", covReturns)

#compute initial portfolio
mvo_weights = compute_mvo_weights(meanReturns, covReturns, stock_dimension) #use pyportfolioOpt to compute mvo weights
lastPrice = np.array([1/p for p in stockData.tail(1).to_numpy()[0]])
initial_portfolio = np.multiply(mvo_weights, lastPrice)

portfolio_assets = tradeData @ initial_portfolio
mvo_result = pd.DataFrame(portfolio_assets, columns=["mean variance mvo"])


# ----DIJA:
TRAIN_START_DATE = '2009-01-02'
TRAIN_END_DATE = '2020-06-30'
TEST_START_DATE = '2020-07-01'
TEST_END_DATE = '2021-10-27'

df_dji = YahooDownloader(
    start_date = TEST_START_DATE,
    end_date = TEST_END_DATE,
    ticker_list=['dji']
).fetch_data()

df_dji = df_dji[['date', 'close']]
fst_day = df_dji['close'][0]

# FIXME: don't know what this is doing
dji = pd.merge(
    df_dji['date'], 
    df_dji['close'].div(fst_day).mul(1000000),
    how='outer',
    left_index=True,
    right_index=True
).set_index('date')

# ---------compare results
# backtest results
df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])

result=pd.DataFrame()

result=pd.merge(result, df_result_a2c, how='outer', left_index=True, right_index=True)
result=pd.merge(result, df_result_ppo, how='outer', left_index=True, right_index=True, suffixes=('', '_drop'))

result = pd.merge(result, mvo_result, how='outer', left_index=True, right_index=True)
result = pd.merge(result, dji, how='outer', left_index=True, right_index=True).fillna(method='bfill')

col_name = []
col_name.append('A2C')
col_name.append('PPO')
col_name.append('mean variance mvo')
col_name.append('DIJA')
result.columns=col_name

print(result)

save_dir='results/graphs'
os.makedirs(save_dir, exist_ok=True)

plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot()
plt.savefig(os.path.join(save_dir, "backttest_result.png"))
plt.close()