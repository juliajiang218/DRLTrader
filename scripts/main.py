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

from preprocessor.yahoodownloader import YahooDownloader
from EnvTrade.env_trading import StockTradingEnv
from agents.DRLAgent import DRLAgent
# from agents.DRLEnsembleAgent import DRLEnsembleAgent
# from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
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
    TEST_END_DATE
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
check_and_make_directories([
    DATA_SAVE_DIR, 
    TRAINED_MODEL_DIR, 
    TENSORBOARD_LOG_DIR, 
    RESULTS_DIR
])


# =====================
# Data Loading & Preprocessing
# =====================
def preprocess_data(is_parquet=True):
    """Load and preprocess training and test data from CSV or Parquet files."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for train data - try CSV first, then Parquet
    train_csv_path = os.path.join(base_path, 'datasets', 'train_data.csv')
    train_parquet_path = os.path.join(base_path, 'datasets', 'train.parquet')
    
    if os.path.exists(train_csv_path) and not is_parquet:
        train = pd.read_csv(train_csv_path)
        logger.info(f"Loaded train data from CSV: {train_csv_path}")
    elif os.path.exists(train_parquet_path) and is_parquet:
        train = pd.read_parquet(train_parquet_path)
        logger.info(f"Loaded train data from Parquet: {train_parquet_path}")
    else:
        raise FileNotFoundError(f"Train data not found. Checked: {train_csv_path} and {train_parquet_path}")
    
    # Check for test data - try CSV first, then Parquet
    test_csv_path = os.path.join(base_path, 'datasets', 'trade_data.csv')
    test_parquet_path = os.path.join(base_path, 'datasets', 'test.parquet')
    
    if os.path.exists(test_csv_path) and not is_parquet:
        test = pd.read_csv(test_csv_path)
        logger.info(f"Loaded test data from CSV: {test_csv_path}")
    elif os.path.exists(test_parquet_path) and is_parquet:
        test = pd.read_parquet(test_parquet_path)
        logger.info(f"Loaded test data from Parquet: {test_parquet_path}")
    else:
        raise FileNotFoundError(f"Test data not found. Checked: {test_csv_path} and {test_parquet_path}")

    # Convert date columns to datetime - use 'Date' if present, else 'date'
    date_col = 'Date' if 'Date' in train.columns else 'date'
    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col] = pd.to_datetime(test[date_col])
    
    # Get actual start and end dates from the datasets
    actual_train_start = train[date_col].min()
    actual_train_end = train[date_col].max()
    actual_test_start = test[date_col].min()
    actual_test_end = test[date_col].max()
    
    print(f"CONFIGURED TRAIN PERIOD: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"ACTUAL TRAIN DATA RANGE: {actual_train_start.strftime('%Y-%m-%d')} to {actual_train_end.strftime('%Y-%m-%d')}")
    print()
    print(f"CONFIGURED TEST PERIOD:  {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"ACTUAL TEST DATA RANGE:  {actual_test_start.strftime('%Y-%m-%d')} to {actual_test_end.strftime('%Y-%m-%d')}")
    print()
    df = pd.concat([train, test])
    df = df.set_index('Day')
    train = train.set_index('Day') 
    test = test.set_index('Day')
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"DF:\n{df}")
    return df, train, test

# =====================
# Environment Setup
# =====================
def create_env_kwargs(df):
    """Create environment keyword arguments based on dataframe and config."""
    stock_dimension = len(df.Stock.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000, # 1 Million
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
def train_a2c_agent(agent, iteration):
    """Train and save an A2C agent."""
    logger.info("Training A2C agent with iteration: " + iteration)
    A2C_PARAMS = {
        "n_steps": 16, 
        "ent_coef": 0.1,
        "learning_rate": 0.00005,
        "max_grad_norm":0.1,  
        "vf_coef":0.25
    }
    # PART 1:
    model_a2c = agent.get_model(
        "a2c", 
        model_kwargs=A2C_PARAMS,
        tensorboard_log=TENSORBOARD_LOG_DIR + "/a2c" + iteration
    )
    tmp_path = RESULTS_DIR + "/a2c" + iteration
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_a2c.set_logger(new_logger_a2c)
    
    # PART 2:
    trained_a2c = agent.train_model(
        model=model_a2c, 
        tb_log_name="a2c",  #FIXME: IS THIS SAVED? WHAT DOES THIS DO??
        total_timesteps=10_000_000
    ) # 10 million 
    trained_a2c.save(TRAINED_MODEL_DIR + "/AGENT_a2c" + iteration)

    logger.info("A2C agent trained and saved.")

def train_ppo_agent(agent, iteration):
    """Train and save a PPO agent."""
    logger.info("Training PPO agent with iteration: " + iteration)
    PPO_PARAMS = {
        "n_steps": 2048, 
        "ent_coef": 0.001, #reduced from 0.1
        "learning_rate": 0.0001, #reduced from 0.00025
        "batch_size": 256, #doubled
        "n_epochs": 10, # more training per batch
        "gae_lambda": 0.95, # better advantage estimation
        "clip_range": 0.1,  #conservative policy updates
    }
    model = agent.get_model(
        "ppo", 
        model_kwargs=PPO_PARAMS, 
        tensorboard_log=TENSORBOARD_LOG_DIR + "/ppo" + iteration
    )
    tmp_path = RESULTS_DIR + "/ppo" + iteration

    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    trained_model = agent.train_model(
        model=model, 
        tb_log_name="ppo", 
        total_timesteps=10_000_000)  
    trained_model.save(TRAINED_MODEL_DIR + "/AGENT_ppo" + iteration)
    logger.info("PPO agent trained and saved.")

def train_ddpg_agent(agent, iteration):
    """Train and save a DDPG agent."""
    logger.info("Training DDPG agent with iteration: " + iteration)
    model = agent.get_model("ddpg", tensorboard_log=TENSORBOARD_LOG_DIR + "/ddpg" + iteration)
    tmp_path = RESULTS_DIR + "/ddpg" + iteration
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    trained_model = agent.train_model(
        model=model, 
        tb_log_name="ddpg", 
        total_timesteps=10_000_000 )  
    save_path = os.path.join(TRAINED_MODEL_DIR, "AGENT_ddpg" + iteration)
    trained_model.save(save_path)
    logger.info(f"DDPG agent trained and saved to {save_path}.")

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
    rebalance_window = 63 # 3 months
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
    timesteps_dict = {'a2c': 10000, 'ppo': 10000, 'ddpg': 10000}
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
    """
    Main function to orchestrate data loading, environment setup, and agent training.
    
    Note: 
    Individual Algorithms must be trained with "train" dataset, while ensemble agent is trained with the entire dataset.
    """
    df, train, test = preprocess_data()
    
    print(df.columns.tolist())
    env_kwargs, stock_dimension, state_space = create_env_kwargs(train)
    # # print("Stock dimension:", stock_dimension)
    # # print("State space:", state_space)

    # # it needs to be "train" for a2c, ppo, ddpg
    train_env = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = train_env.get_sb_env()
    agent = DRLAgent(env=env_train)

    iteration = '0822_dataset2'
    # train_a2c_agent(agent, iteration)
    # train_ppo_agent(agent, iteration)
    train_ddpg_agent(agent, iteration)
    # # train_ensemble_agent(df, stock_dimension, state_space)
    # # preprocess_data

# =====================
# Script Entry Point
# =====================
if __name__ == "__main__":
    main()
    # papertrade()