# =====================
# Imports and Logging
# =====================
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import logging
import datetime
import itertools
from pprint import pprint

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import configure

# Project-specific imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../FinRL-Library")

from finrl.config_tickers import DOW_30_TICKER
from preprocessor.yahoodownloader import YahooDownloader
from env_stock_trading.env_stocktrading import StockTradingEnv
from agents.DRLAgent import DRLAgent
from agents.DRLEnsembleAgent import DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

# =====================
# Logging Configuration
# =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =====================
# Constants & Directories
# =====================
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# --- Original date ranges ---
# TRAIN_START_DATE = '2010-01-01'
# TRAIN_END_DATE = '2021-10-01'
# TEST_START_DATE = '2021-10-01'
# TEST_END_DATE = '2023-03-01'

# --- Custom/active date ranges ---
TRAIN_START_DATE = '2009-01-02'
TRAIN_END_DATE = '2020-06-30'
TEST_START_DATE = '2020-07-01'
TEST_END_DATE = '2021-10-27'

# =====================
# Data Loading & Preprocessing
# =====================
def preprocess_data():
    """Load and preprocess training and test data from CSV files."""
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'train_data.csv')
    test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'trade_data.csv')
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    print("Train Dataset: \n", train[train['date'].isin(train['date'].unique()[:2].tolist())] )
    df = pd.concat([train, test])
    df = df.set_index(df.columns[0])
    df.index.names = ['']
    train = train.set_index(train.columns[0])
    train.index.names = ['']
    test = test.set_index(test.columns[0])
    test.index.names = ['']
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df, train, test

# =====================
# Environment Setup
# =====================
def create_env_kwargs(df):
    """Create environment keyword arguments based on dataframe and config."""
    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    logger.info(f"Environment state_space: {state_space}, stock_dim: {stock_dimension}")
    return env_kwargs, stock_dimension, state_space

# =====================
# Agent Training Functions
# =====================
def train_a2c_agent(agent):
    """Train and save an A2C agent."""
    logger.info("Training A2C agent...")
    model_a2c = agent.get_model("a2c")
    tmp_path = RESULTS_DIR + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv"]) # , "tensorboard"
    model_a2c.set_logger(new_logger_a2c)
    trained_a2c = agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=10000000) 
    trained_a2c.save(TRAINED_MODEL_DIR + "/AGENT_a2c")
    logger.info("A2C agent trained and saved.")

def train_ppo_agent(agent):
    """Train and save a PPO agent."""
    logger.info("Training PPO agent...")
    model = agent.get_model("ppo")
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    trained_model = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=50000)
    trained_model.save(TRAINED_MODEL_DIR + "/AGENT_ppo")
    logger.info("PPO agent trained and saved.")

def train_ddpg_agent(agent):
    """Train and save a DDPG agent."""
    logger.info("Training DDPG agent...")
    model = agent.get_model("ddpg")
    tmp_path = RESULTS_DIR + "/ddpg"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    trained_model = agent.train_model(model=model, tb_log_name="ddpg", total_timesteps=50000)
    trained_model.save(TRAINED_MODEL_DIR + "/AGENT_ddpg")
    logger.info("DDPG agent trained and saved.")

def train_ensemble_agent(df, stock_dimension, state_space):
    """Train and run an ensemble agent."""
    logger.info("Training Ensemble agent...")
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 5
    }
    rebalance_window = 63
    validation_window = 63
    ensemble_agent = DRLEnsembleAgent(
        df=df,
        train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
        val_test_period=(TEST_START_DATE, TEST_END_DATE),
        rebalance_window=rebalance_window,
        validation_window=validation_window,
        **env_kwargs
    )
    A2C_model_kwargs = {'n_steps': 5, 'ent_coef': 0.005, 'learning_rate': 0.0007}
    PPO_model_kwargs = {"ent_coef": 0.01, "n_steps": 2048, "learning_rate": 0.00025, "batch_size": 128}
    DDPG_model_kwargs = {"buffer_size": 10_000, "learning_rate": 0.0005, "batch_size": 128}
    timesteps_dict = {'a2c': 50_000, 'ppo': 50_000, 'ddpg': 50_000}
    df_summary = ensemble_agent.run_ensemble_strategy(
        A2C_model_kwargs=A2C_model_kwargs,
        PPO_model_kwargs=PPO_model_kwargs,
        DDPG_model_kwargs=DDPG_model_kwargs,
        timesteps_dict=timesteps_dict
    )
    logger.info("Ensemble agent training complete.")
    return df_summary

# =====================
# Main Orchestration
# =====================
def main():
    """Main function to orchestrate data loading, environment setup, and agent training."""
    df, train, test = preprocess_data()
    
    # print(df)
    env_kwargs, stock_dimension, state_space = create_env_kwargs(df)
    # print("Stock dimension:", stock_dimension)
    # print("State space:", state_space)

    train_env = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = train_env.get_sb_env()
    agent = DRLAgent(env=env_train)

    train_a2c_agent(agent)
    # train_ppo_agent(agent)
    # train_ddpg_agent(agent)
    # train_ensemble_agent(df, stock_dimension, state_space)

# =====================
# Script Entry Point
# =====================
if __name__ == "__main__":
    main()