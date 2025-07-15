"""
Disclaimer:
This file contains the DRLAgent class, which is a custom agent for training reinforcement learning agents to trade stocks.
Code: This implementation borrows code from https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/agent/drl_agent.py

It is used to train the agents in the DRLAgent class. 
For educational purposes only.
"""
from __future__ import annotations

import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import statistics
import time
from typing import Type, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config
from env_stock_trading import StockTradingEnv
# from preprocessor.preprocessors import data_split

# predefined models, the agent can only be trained with any of these models
MODELS = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

# noise types for exploration
NOISE = {
    "normal": NormalActionNoise, # normal noise (random Gaussian noise)
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise, # correlated noise tends to return to mean
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.

    Tracks the reward, min, mean, and max reward.
    Handles the case where "rewards" is not found.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

    def _on_rollout_end(self) -> bool:
        try:
            rollout_buffer_rewards = self.locals["rollout_buffer"].rewards.flatten()
            self.logger.record(
                key="train/reward_min", value=min(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_mean", value=statistics.mean(rollout_buffer_rewards)
            )
            self.logger.record(
                key="train/reward_max", value=max(rollout_buffer_rewards)
            )
        except BaseException as error:
            # Handle the case where "rewards" is not found
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
            print("Logging Error:", error)
        return True


class DebugCallback(BaseCallback):
    """
    Debug callback to trace through episode iterations.
    Prints detailed information about each step.
    """
    
    def __init__(self, verbose=1, max_steps_to_print=100):
        super().__init__(verbose)
        self.step_count = 0
        self.max_steps_to_print = max_steps_to_print
        
    def _on_step(self) -> bool:
        if self.step_count < self.max_steps_to_print:
            # Get current observation/state
            obs = self.locals.get("obs", None)
            
            # Get action taken
            actions = self.locals.get("actions", None)
            
            # Get reward
            rewards = self.locals.get("rewards", None)
            
            # Get done status
            dones = self.locals.get("dones", None)
            
            print(f"Step {self.step_count}:")
            print(f"  Observation shape: {obs.shape if obs is not None else 'None'}")
            print(f"  Actions: {actions}")
            print(f"  Rewards: {rewards}")
            print(f"  Done: {dones}")
            print("-" * 50)
            
        self.step_count += 1
        return True
        
    def _on_rollout_end(self) -> bool:
        print(f"Rollout ended after {self.step_count} steps")
        print("=" * 50)
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        """
        Initialize any DRL algorithm. 
        Set up neural network policy="MlpPolicy", 
        configure action-noise for exploration. 
        Returns ready-to-train model.
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        # this if condition handles exploration noise for continuous action spaces
        # add noise helps the agent explore different strategies during training
        # without noise, agent might stuck in local optima
        # 10% nosie level relative to action space
        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1] # determines how many continuous actions the agent can take
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            ) # noise type: normal noise (random Gaussian noise), Ornstein-Uhlenbeck noise (correlated noise tends to return to mean)
        print(model_kwargs)

        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod # avoids creating a DRLAgent instance
    def train_model(
        model,
        tb_log_name,
        total_timesteps, #deleted default timesteps of 5000
        callbacks: Optional[List[BaseCallback]] = None,
    ):  # this function is static method, so it can be called without creating an instance of the class
        """
        Training Wrapper, calls model.learn() with proper logging and callback setup.

        Train the model, automatically logs to tensorboard, 
        use static method to avoid creating an instance of the class.
        """
        # model.learn() is defined in stable_baselines3
        # it's a method that all models inherit from base classes.
        # this learn method handles the actual reinforcement learning training loop
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            # TensorboardCallback: logs reward stats to TensorBoard for visualization
            # Custom callbacks: can be passed in to add additional monitoring/logging
            callback=(
                CallbackList(
                    [TensorboardCallback()] + callbacks
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """
        Makes a trading prediction on test data.
        Deterministic = True, means the model will always make the same action for the same state.
        Steps through each trading day in the dataset.
        Saves account value and trading actions for analysis.
        Returns memory of account performance, and actions taken.
        """
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps - 1
            ):  # more descriptive condition for early termination to clarify the logic
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0] if account_memory is not None else None, actions_memory[0] if actions_memory is not None else None

    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        "Loads pretrained model from filepath cwd, runs full trading simulation, tracks total assets over time, returns episode performance data."
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        # test on the testing env
        state = environment.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = (
                environment.amount
                + (environment.price_ary[environment.day] * environment.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset
            episode_returns.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")
        return episode_total_assets

    @staticmethod
    def trace_episode(model, environment, deterministic=True, max_steps=None):
        """
        Trace through a single episode with detailed logging.
        Useful for debugging and understanding model behavior.
        
        Args:
            model: Trained model
            environment: Trading environment
            deterministic: Whether to use deterministic actions
            max_steps: Maximum steps to trace (None for full episode)
        """
        print("Starting episode trace...")
        print("=" * 60)
        
        # Get environment
        test_env, test_obs = environment.get_sb_env()
        test_env.reset()
        
        step_count = 0
        total_reward = 0
        
        # Get max steps
        if max_steps is None:
            max_steps = len(environment.df.index.unique()) - 1
        
        print(f"Episode will run for maximum {max_steps} steps")
        print("-" * 60)
        
        for i in range(max_steps):
            # Get action from model
            action, _states = model.predict(test_obs, deterministic=deterministic)
            
            # Take step in environment
            test_obs, rewards, dones, info = test_env.step(action)
            
            # Log step information
            print(f"Step {step_count}:")
            print(f"  Action: {action}")
            print(f"  Reward: {rewards}")
            print(f"  Done: {dones}")
            print(f"  Info: {info}")
            
            # Try to get account value if available
            try:
                account_value = test_env.env_method(method_name="get_account_value")
                print(f"  Account Value: {account_value}")
            except:
                print(f"  Account Value: Not available")
            
            print("-" * 40)
            
            step_count += 1
            total_reward += rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
            
            if dones[0]:
                print("Episode ended early!")
                break
        
        print("=" * 60)
        print(f"Episode completed in {step_count} steps")
        print(f"Total reward: {total_reward}")
        print("=" * 60)
        
        return step_count, total_reward

def demo_a2c_training():
    """
    Demo function to show A2C training with 3 stocks and 2 technical indicators.
    Creates synthetic data and demonstrates the complete training pipeline.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print("=" * 60)
    print("A2C Training Demo with 3 Stocks and 2 Technical Indicators")
    print("=" * 60)
    
    # Create synthetic stock data for 3 stocks in long format with required columns
    np.random.seed(42)  # For reproducible results
    from itertools import product

    # Define stocks and date range
    stocks = ['AAPL', 'GOOGL', 'MSFT']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    # Prepare data rows
    data_rows = []
    for day_idx, date in enumerate(dates):
        for tic in stocks:
            # Synthetic price and indicator generation
            base_price = {'AAPL': 100, 'GOOGL': 50, 'MSFT': 75}[tic]
            day_num = (date - start_date).days
            # Simulate price as random walk
            close = base_price * (1 + np.random.normal(0.001, 0.02) * day_num / n_days)
            open_ = close * np.random.uniform(0.98, 1.02)
            high = max(open_, close) * np.random.uniform(1.00, 1.03)
            low = min(open_, close) * np.random.uniform(0.97, 1.00)
            volume = np.random.randint(100000, 1000000)
            macd = np.random.normal(0, 1)
            rsi_30 = np.random.uniform(10, 90)
            turbulence = np.random.uniform(0, 100)
            # Only keep two indicators: macd, rsi_30
            data_rows.append({
                'date': date,
                'tic': tic,
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'day': day_num,
                'macd': macd,
                'rsi_30': rsi_30,
                'turbulence': turbulence
            })

    # Create DataFrame and set proper index
    df = pd.DataFrame(data_rows)
    # Set index to be the same for all stocks on the same date
    df['date_idx'] = df.groupby('date').ngroup()
    df = df.set_index('date_idx')
    print(f"Created synthetic data for {len(df)} rows, {len(dates)} unique dates, {len(stocks)} stocks")
    print("Columns:", df.columns.tolist())
    print("Index levels:", df.index.names)
    print(df.head(10))

    # Environment parameters
    stock_dim = 3  # 3 stocks
    hmax = 100  # Maximum shares to hold
    initial_amount = 10000  # Initial capital
    num_stock_shares = [0, 0, 0]  # Initial shares for each stock
    buy_cost_pct = [0.001, 0.001, 0.001]  # 0.1% buy commission for each stock
    sell_cost_pct = [0.001, 0.001, 0.001]  # 0.1% sell commission for each stock
    tech_indicator_list = ['macd', 'rsi_30']
    reward_scaling = 1e-4  # Scale rewards
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
    
    env_kwargs = {
        "hmax": hmax, # max shares to hold
        "initial_amount": initial_amount, # Initial capital
        "num_stock_shares": num_stock_shares, # Initial shares for each stock
        "buy_cost_pct": buy_cost_pct, # 0.1% buy commission for each stock
        "sell_cost_pct": sell_cost_pct, #0.1% sell commission for each stock
        "state_space": state_space, #
        "stock_dim": stock_dim,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dim,
        "reward_scaling": reward_scaling
    }
   
    # Create environment
    try:
        env = StockTradingEnv(df=df, **env_kwargs) # this creates a StockTradingEnv object
        print("ENV: ", env)
        env_train, obs = env.get_sb_env()

        print("✓ Environment created successfully: \n", env_train)
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return
    
    # Create agent and model
    try:
        agent = DRLAgent(env)
        model = agent.get_model("a2c", verbose=1)
        print("✓ A2C model created successfully")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return
    
    # Train the model
    print(f"\nStarting A2C training...")
    print("-" * 40)
    
    try:
        # Train with debug callback to see first few steps
        trained_model = DRLAgent.train_model(
            model=model,
            tb_log_name="a2c_demo",
            total_timesteps=1000,  # Small number for demo
            callbacks=[DebugCallback(max_steps_to_print=10)]  # Show first 10 steps
        )
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return
    
    # Test the trained model
    print(f"\nTesting trained model...")
    print("-" * 40)
    
    try:
        # Trace through a short episode
        steps, total_reward = DRLAgent.trace_episode(
            model=trained_model,
            environment=env,
            deterministic=True,
            max_steps=1000 
        )
        print(f"✓ Episode trace completed: {steps} steps, total reward: {total_reward:.6f}")
    except Exception as e:
        print(f"✗ Error during testing: {e}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the demo
    demo_a2c_training()
    