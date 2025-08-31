"""
Disclaimer:
This file contains the DRLEnsembleAgent class, which is a custom ensemble agent for training reinforcement learning agents to trade stocks.
Code: This implementation borrows code from https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/agents/stablebaselines3/models.py

It is used to train the agents in the DRLAgent class. The script is modified to train agents in the customized DRLAgent class.

Bug Fix (2024-07-25):
- Logging in callbacks now checks for 'rollout_buffer' (on-policy algorithms like PPO/A2C) or 'replay_buffer' (off-policy algorithms like DDPG/TD3/SAC).
- This prevents errors and ensures correct reward logging for all supported algorithms.

For educational purposes only.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

MODELS = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
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
            if hasattr(self.model, "rollout_buffer"):
                # On-policy: PPO, A2C, etc.
                rewards = self.model.rollout_buffer.rewards.flatten()
            elif hasattr(self.model, "replay_buffer"):
                # Off-policy: DDPG, TD3, SAC, etc.
                # You may want to sample or summarize from the replay buffer
                # For example, log the most recent batch of rewards
                rewards = self.model.replay_buffer.rewards.flatten()
                # print(f"Rewards added to ddpg csv: {rewards}\n")
            else:
                rewards = None

            if rewards is not None and len(rewards) > 0:
                self.logger.record("train/reward_min", min(rewards))
                self.logger.record("train/reward_mean", float(sum(rewards)) / len(rewards))
                self.logger.record("train/reward_max", max(rewards))
            else:
                self.logger.record("train/reward_min", None)
                self.logger.record("train/reward_mean", None)
                self.logger.record("train/reward_max", None)
        except BaseException as error:
            # Handle the case where "rewards" is not found
            self.logger.record(key="train/reward_min", value=None)
            self.logger.record(key="train/reward_mean", value=None)
            self.logger.record(key="train/reward_max", value=None)
            print("Logging Error:", error)
        return True


class DRLEnsembleAgent:
    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        seed=None,
        verbose=1,
        iteration=None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[
                temp_model_kwargs["action_noise"]
            ](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        print(temp_model_kwargs)
        # Create iteration-specific tensorboard log directory
        if iteration is not None:
            tensorboard_path = f"{config.TENSORBOARD_LOG_DIR}/{model_name}_{iteration}"
        else:
            tensorboard_path = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
            
        return MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=tensorboard_path,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )

    @staticmethod
    def train_model(
        model,
        model_name,
        tb_log_name,
        iter_num,
        total_timesteps=5000,
        callbacks: Type[BaseCallback] = None,
    ):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=(
                CallbackList(
                    [TensorboardCallback()] + [callback for callback in callbacks]
                )
                if callbacks is not None
                else TensorboardCallback()
            ),
        )
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}"
        )
        return model

    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """Calculate Sharpe ratio based on validation results"""
        df_total_value = pd.read_csv(
            f"results/account_value_validation_{model_name}_{iteration}.csv"
        )
        # If the agent did not make any transaction
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )

    @staticmethod
    def calculate_directional_accuracy(iteration, model_name):
        """Calculate directional accuracy - how often the model predicts price direction correctly"""
        try:
            # Load actions and account values
            df_actions = pd.read_csv(f"results/actions_validation_{model_name}_{iteration}.csv")
            df_account = pd.read_csv(f"results/account_value_validation_{model_name}_{iteration}.csv")
            
            # Calculate forward returns for each day
            df_account['forward_return'] = df_account['account_value'].pct_change().shift(-1)
            
            # Calculate portfolio weight changes as a proxy for predicted direction
            # Sum absolute actions as overall directional signal
            portfolio_signal = df_actions.iloc[:, 1:].sum(axis=1)  # Skip date column
            actual_returns = df_account['forward_return'].iloc[:-1]  # Drop last NaN
            
            # Align lengths
            min_len = min(len(portfolio_signal), len(actual_returns))
            portfolio_signal = portfolio_signal.iloc[:min_len]
            actual_returns = actual_returns.iloc[:min_len]
            
            # Calculate directional accuracy
            predicted_direction = np.sign(portfolio_signal)
            actual_direction = np.sign(actual_returns)
            
            # Remove zero directions (no clear signal)
            valid_mask = (predicted_direction != 0) & (actual_direction != 0)
            if valid_mask.sum() == 0:
                return 0.5  # Random chance if no valid signals
                
            directional_accuracy = (predicted_direction[valid_mask] == actual_direction[valid_mask]).mean()
            return directional_accuracy
            
        except Exception as e:
            print(f"Error calculating directional accuracy for {model_name}_{iteration}: {e}")
            return 0.5  # Return random chance on error
    
    @staticmethod
    def calculate_information_coefficient(iteration, model_name):
        """Calculate Information Coefficient - correlation between predictions and future returns"""
        try:
            # Load actions and account values
            df_actions = pd.read_csv(f"results/actions_validation_{model_name}_{iteration}.csv")
            df_account = pd.read_csv(f"results/account_value_validation_{model_name}_{iteration}.csv")
            
            # Calculate forward returns
            df_account['forward_return'] = df_account['account_value'].pct_change().shift(-1)
            
            # Use sum of absolute actions as prediction signal
            portfolio_signal = df_actions.iloc[:, 1:].sum(axis=1)  # Skip date column
            actual_returns = df_account['forward_return'].iloc[:-1]  # Drop last NaN
            
            # Align lengths
            min_len = min(len(portfolio_signal), len(actual_returns))
            portfolio_signal = portfolio_signal.iloc[:min_len]
            actual_returns = actual_returns.iloc[:min_len]
            
            # Remove NaN and infinite values
            valid_mask = np.isfinite(portfolio_signal) & np.isfinite(actual_returns)
            if valid_mask.sum() < 2:
                return 0.0  # Need at least 2 points for correlation
                
            clean_signals = portfolio_signal[valid_mask]
            clean_returns = actual_returns[valid_mask]
            
            # Calculate correlation (Information Coefficient)
            if len(clean_signals) < 2 or clean_signals.std() == 0 or clean_returns.std() == 0:
                return 0.0
                
            ic = np.corrcoef(clean_signals, clean_returns)[0, 1]
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            print(f"Error calculating IC for {model_name}_{iteration}: {e}")
            return 0.0  # Return 0 on error

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
    ):
        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.train_env = None  # defined in train_validation() function

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """validation process"""
        for _ in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(
        self, model, name, last_state, iter_num, turbulence_threshold, initial
    ):
        """make a prediction based on trained model"""

        # trading env
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=trade_data,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    initial=initial,
                    previous_state=last_state,
                    model_name=name,
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = trade_env.envs[0].render()

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_{name}_{i}.csv", index=False)
        return last_state

    def _train_window(
        self,
        model_name,
        model_kwargs,
        sharpe_list,
        validation_start_date,
        validation_end_date,
        timesteps_dict,
        i,
        validation,
        turbulence_threshold,
    ):
        """
        Train the model for a single window.
        """
        if model_kwargs is None:
            return None, sharpe_list, -1, 0.5, 0.0

        print(f"======{model_name} Training========")
        model = self.get_model(
            model_name, self.train_env, policy="MlpPolicy", model_kwargs=model_kwargs, iteration=i
        )
        model = self.train_model(
            model,
            model_name,
            tb_log_name=f"{model_name}_{i}",
            iter_num=i,
            total_timesteps=timesteps_dict[model_name],
        )  # 100_000
        print(
            f"======{model_name} Validation from: ",
            validation_start_date,
            "to ",
            validation_end_date,
        )
        val_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=validation,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    iteration=i,
                    model_name=model_name,
                    mode="validation",
                    print_verbosity=self.print_verbosity,
                )
            ]
        )
        val_obs = val_env.reset()
        self.DRL_validation(
            model=model,
            test_data=validation,
            test_env=val_env,
            test_obs=val_obs,
        )
        
        # Calculate multiple performance metrics
        sharpe = self.get_validation_sharpe(i, model_name=model_name)
        directional_accuracy = self.calculate_directional_accuracy(i, model_name=model_name)
        ic = self.calculate_information_coefficient(i, model_name=model_name)
        
        print(f"{model_name} Sharpe Ratio: {sharpe:.4f}")
        print(f"{model_name} Directional Accuracy: {directional_accuracy:.4f}")
        print(f"{model_name} Information Coefficient: {ic:.4f}")
        
        sharpe_list.append(sharpe)
        return model, sharpe_list, sharpe, directional_accuracy, ic

    def run_ensemble_strategy(
        self,
        A2C_model_kwargs,
        PPO_model_kwargs,
        DDPG_model_kwargs,
        timesteps_dict
    ):
        # Model Parameters
        kwargs = {
            "a2c": A2C_model_kwargs,
            "ppo": PPO_model_kwargs,
            "ddpg": DDPG_model_kwargs
  
        }
        # Model Performance Metrics
        model_dct = {k: {
            "sharpe_list": [], "sharpe": -1,
            "directional_accuracy_list": [], "directional_accuracy": 0.5,
            "ic_list": [], "ic": 0.0
        } for k in MODELS.keys()}

        """Ensemble Strategy that combines A2C, PPO, DDPG"""
        print("============Start Ensemble Strategy============")
        # for ensemble model, it's necessary to feed the last state
        # of the previous model to the current model as the initial state
        last_state_ensemble = []

        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[
            (self.df.date < self.train_period[1])
            & (self.df.date >= self.train_period[0])
        ]
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            # initial state is empty
            if i - self.rebalance_window - self.validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter (63 days)
            end_date_index = self.df.index[
                self.df["date"]
                == self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ]
            ].to_list()[-1]
            start_date_index = end_date_index - 63 + 1

            historical_turbulence = self.df.iloc[
                start_date_index : (end_date_index + 1), :
            ]

            historical_turbulence = historical_turbulence.drop_duplicates(
                subset=["date"]
            )

            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )

            # print(historical_turbulence_mean)

            if historical_turbulence_mean > insample_turbulence_threshold:
                # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_turbulence_threshold
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )
            print("turbulence_threshold: ", turbulence_threshold)

            # Environment Setup starts
            # training env
            train = data_split(
                self.df,
                start=self.train_period[0],
                end=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            self.train_env = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        df=train,
                        stock_dim=self.stock_dim,
                        hmax=self.hmax,
                        initial_amount=self.initial_amount,
                        num_stock_shares=[0] * self.stock_dim,
                        buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                        sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                        reward_scaling=self.reward_scaling,
                        state_space=self.state_space,
                        action_space=self.action_space,
                        tech_indicator_list=self.tech_indicator_list,
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )

            validation = data_split(
                self.df,
                start=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
                end=self.unique_trade_date[i - self.rebalance_window],
            )
            # Environment Setup ends

            # Training and Validation starts
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
            # print("==============Model Training===========")
            # Train Each Model
            for model_name in MODELS.keys():
                # Train The Model
                model, sharpe_list, sharpe, directional_accuracy, ic = self._train_window(
                    model_name,
                    kwargs[model_name],
                    model_dct[model_name]["sharpe_list"],
                    validation_start_date,
                    validation_end_date,
                    timesteps_dict,
                    i,
                    validation,
                    turbulence_threshold,
                )
                # Save all performance metrics
                model_dct[model_name]["sharpe_list"] = sharpe_list
                model_dct[model_name]["model"] = model
                model_dct[model_name]["sharpe"] = sharpe
                
                # Store additional metrics
                model_dct[model_name]["directional_accuracy_list"].append(directional_accuracy)
                model_dct[model_name]["directional_accuracy"] = directional_accuracy
                model_dct[model_name]["ic_list"].append(ic)
                model_dct[model_name]["ic"] = ic

            print(
                "======Best Model Retraining from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[i - self.rebalance_window],
            )
            # Environment setup for model retraining up to first trade date
            # train_full = data_split(self.df, start=self.train_period[0],
            # end=self.unique_trade_date[i - self.rebalance_window])
            # self.train_full_env = DummyVecEnv([lambda: StockTradingEnv(train_full,
            #                                               self.stock_dim,
            #                                               self.hmax,
            #                                               self.initial_amount,
            #                                               self.buy_cost_pct,
            #                                               self.sell_cost_pct,
            #                                               self.reward_scaling,
            #                                               self.state_space,
            #                                               self.action_space,
            #                                               self.tech_indicator_list,
            #                                              print_verbosity=self.print_verbosity
            # )])
            # Model Selection based on sharpe ratio
            # Same order as MODELS: {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
            sharpes = [model_dct[k]["sharpe"] for k in MODELS.keys()]
            # Find the model with the highest sharpe ratio
            max_mod = list(MODELS.keys())[np.argmax(sharpes)]
            model_use.append(max_mod.upper())
            model_ensemble = model_dct[max_mod]["model"]
            # Training and Validation ends

            # Trading starts
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            # print("Used Model: ", model_ensemble)
            last_state_ensemble = self.DRL_prediction(
                model=model_ensemble,
                name="ensemble",
                last_state=last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            )
            # Trading ends

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                model_use,
                model_dct["a2c"]["sharpe_list"],
                model_dct["ppo"]["sharpe_list"],
                model_dct["ddpg"]["sharpe_list"],
                model_dct["a2c"]["directional_accuracy_list"],
                model_dct["ppo"]["directional_accuracy_list"],
                model_dct["ddpg"]["directional_accuracy_list"],
                model_dct["a2c"]["ic_list"],
                model_dct["ppo"]["ic_list"],
                model_dct["ddpg"]["ic_list"]
            ]
        ).T
        df_summary.columns = [
            "Iter",
            "Val Start",
            "Val End",
            "Model Used",
            "A2C Sharpe",
            "PPO Sharpe",
            "DDPG Sharpe",
            "A2C Directional Accuracy",
            "PPO Directional Accuracy",
            "DDPG Directional Accuracy",
            "A2C Information Coefficient",
            "PPO Information Coefficient",
            "DDPG Information Coefficient"
        ]

        return df_summary