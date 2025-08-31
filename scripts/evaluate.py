"""
How to run this file:

python3 -m venv ~/yfinance
source ~/yfinance/bin/activate
pip3 install yfinance

"""

import datetime
from datetime import datetime
from main import preprocess_data, create_env_kwargs
from stable_baselines3 import A2C, DDPG, PPO
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
import sys, os
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import numpy as np

# allows to import modules/packages from project root and its subdirectories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # without this, python can't find modules located outside main.py directory
from preprocessor.yahoodownloader import YahooDownloader
from EnvTrade.env_trading import StockTradingEnv
from agents.DRLAgent import DRLAgent

# load test dataset
_, train, test = preprocess_data()

# load trained models
trained_a2c = A2C.load("/deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/trained_models/AGENT_a2c.zip")
trained_ppo = PPO.load("/deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/trained_models/AGENT_ppo.zip")
# trained_ddpg = DDPG.load("/deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/trained_models/AGENT_ddpg.zip")

# use trained models to trade on test dataset
    # create trading environment, based on test dataset
env_kwargs, stock_dimension, state_space = create_env_kwargs(test)
test_env = StockTradingEnv(
    df=test,
    turbulence_threshold=70, 
    risk_indicator_col='vix', 
    **env_kwargs
)
env_trade, _ = test_env.get_sb_env()

    # start testing
df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c,
    environment= test_env
)

# DRLAgent.trace_episode(
#     model=trained_a2c,
#     environment=test_env,
#     max_steps=365,
#     model_name="A2C"
# )

# DRLAgent.trace_episode(
#     model=trained_ppo,
#     environment=test_env,
#     max_steps=365,
#     model_name="PPO"
# )

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo,
    environment= test_env
)

# Calculate prediction quality metrics using actual stock price movements
def calculate_directional_accuracy(actions_df, test_data):
    """Calculate how often the model's actions align with actual stock price movements"""
    try:
        # Get unique dates from actions
        action_dates = actions_df.iloc[:, 0].values  # First column is date
        action_values = actions_df.iloc[:, 1:].values  # Stock actions
        
        # Get stock tickers (assuming same order as actions)
        stock_tickers = test_data['tic'].unique()
        n_stocks = len(stock_tickers)
        
        correct_predictions = []
        
        for i, date in enumerate(action_dates[:-1]):  # Skip last date (no future return)
            # Get current day's actions for each stock
            current_actions = action_values[i]  # Shape: (n_stocks,)
            
            # Get actual stock returns for next day
            current_date_data = test_data[test_data['date'] == date]
            next_date = action_dates[i + 1] if i + 1 < len(action_dates) else None
            
            if next_date is None:
                continue
                
            next_date_data = test_data[test_data['date'] == next_date]
            
            if len(current_date_data) != n_stocks or len(next_date_data) != n_stocks:
                continue
                
            # Calculate actual returns for each stock
            current_prices = current_date_data.sort_values('tic')['close'].values
            next_prices = next_date_data.sort_values('tic')['close'].values
            actual_returns = (next_prices - current_prices) / current_prices
            
            # Compare directions: action sign vs actual return sign
            predicted_directions = np.sign(current_actions)
            actual_directions = np.sign(actual_returns)
            
            # Count correct predictions (excluding zeros)
            valid_mask = (predicted_directions != 0) & (actual_directions != 0)
            if valid_mask.sum() > 0:
                daily_accuracy = (predicted_directions[valid_mask] == actual_directions[valid_mask]).mean()
                correct_predictions.append(daily_accuracy)
        
        return np.mean(correct_predictions) if correct_predictions else 0.5
        
    except Exception as e:
        print(f"Error calculating directional accuracy: {e}")
        return 0.5

def calculate_information_coefficient(actions_df, test_data):
    """Calculate correlation between model actions and actual stock returns"""
    try:
        action_dates = actions_df.iloc[:, 0].values
        action_values = actions_df.iloc[:, 1:].values
        
        stock_tickers = test_data['tic'].unique()
        n_stocks = len(stock_tickers)
        
        all_actions = []
        all_returns = []
        
        for i, date in enumerate(action_dates[:-1]):
            current_actions = action_values[i]
            
            current_date_data = test_data[test_data['date'] == date]
            next_date = action_dates[i + 1] if i + 1 < len(action_dates) else None
            
            if next_date is None:
                continue
                
            next_date_data = test_data[test_data['date'] == next_date]
            
            if len(current_date_data) != n_stocks or len(next_date_data) != n_stocks:
                continue
            
            # Calculate actual returns
            current_prices = current_date_data.sort_values('tic')['close'].values
            next_prices = next_date_data.sort_values('tic')['close'].values
            actual_returns = (next_prices - current_prices) / current_prices
            
            # Store actions and corresponding returns
            all_actions.extend(current_actions)
            all_returns.extend(actual_returns)
        
        if len(all_actions) < 2:
            return 0.0
            
        all_actions = np.array(all_actions)
        all_returns = np.array(all_returns)
        
        # Remove invalid values
        valid_mask = np.isfinite(all_actions) & np.isfinite(all_returns)
        if valid_mask.sum() < 2:
            return 0.0
            
        clean_actions = all_actions[valid_mask]
        clean_returns = all_returns[valid_mask]
        
        if clean_actions.std() == 0 or clean_returns.std() == 0:
            return 0.0
            
        ic = np.corrcoef(clean_actions, clean_returns)[0, 1]
        return ic if not np.isnan(ic) else 0.0
        
    except Exception as e:
        print(f"Error calculating IC: {e}")
        return 0.0

# Calculate metrics for both models using actual stock price movements
print("\n" + "="*60)
print("MODEL PREDICTION QUALITY EVALUATION")
print("(Comparing model actions vs actual stock price movements)")
print("="*60)

# A2C Metrics
if 'df_actions_a2c' in locals():
    directional_accuracy_a2c = calculate_directional_accuracy(df_actions_a2c, test)
    information_coefficient_a2c = calculate_information_coefficient(df_actions_a2c, test)
    
    print(f"\n=== A2C Model Prediction Quality ===")
    print(f"Directional Accuracy: {directional_accuracy_a2c:.4f} ({directional_accuracy_a2c*100:.2f}%)")
    print(f"Information Coefficient: {information_coefficient_a2c:.4f}")

# PPO Metrics
if 'df_actions_ppo' in locals():
    directional_accuracy_ppo = calculate_directional_accuracy(df_actions_ppo, test)
    information_coefficient_ppo = calculate_information_coefficient(df_actions_ppo, test)
    
    print(f"\n=== PPO Model Prediction Quality ===")
    print(f"Directional Accuracy: {directional_accuracy_ppo:.4f} ({directional_accuracy_ppo*100:.2f}%)")
    print(f"Information Coefficient: {information_coefficient_ppo:.4f}")

print(f"\n=== What These Metrics Mean ===")
print(f"• Directional Accuracy: How often the model's action direction aligns with actual stock price movement")
print(f"• Information Coefficient: Correlation between action magnitude and actual stock returns")
print(f"• Values > 0.5 (accuracy) or > 0 (IC) indicate the model has predictive skill beyond random")
print(f"\nNote: Model predicts ACTIONS (buy/sell signals) not direct prices.")
print(f"Actions are compared against actual next-day stock price movements.")

print(f"\n=== Model Actions Explanation ===")
print(f"The model outputs continuous values [-1, +1] for each of the {test['tic'].nunique()} stocks:")
print(f"• Positive action: Buy signal (larger = stronger buy)")
print(f"• Negative action: Sell signal (more negative = stronger sell)")
print(f"• Near zero: Hold/neutral signal")
print(f"\nActions are used for position sizing in the trading environment.")
print(f"We measure if these signals correctly anticipate stock price movements.")

# Uncomment below to see raw action data:
# print("A2C Actions Sample:")
# print(df_actions_a2c.head())
# print("PPO Actions Sample:") 
# print(df_actions_ppo.head())


# ---------use MVO, DIJA to trade on test dataset
# ----MVO:
#calculate weights for mean-variance
stockData = process_mvo_df(train, stock_dimension)
tradeData = process_mvo_df(test, stock_dimension)

tradeData.to_numpy()

# print(tradeData)

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
# print("Mean returns of assets in k-portfolio \n", meanReturns)
# print("Variance-covariance matrix of returns: \n", covReturns)

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

# add timestep to filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"backtest_result_{timestamp}.png"

plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot()
plt.savefig(os.path.join(save_dir, filename))
plt.close()

print(f"\n=== BENCHMARK COMPARISON RESULTS ===")
print(f"Backtest results saved to: {os.path.join(save_dir, filename)}")
print(result.tail())