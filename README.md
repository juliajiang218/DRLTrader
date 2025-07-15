# Deep Reinforcement Learning Agents with interpretable attribution features in a simulated real-data driven market

## Summary
Integrate Explainable Deep Reinforcement Learning Agents (A2C, PPO, DDPG, Ensemble Strategy) in a simulated real-data driven market to optimize trading strategies and allow users to monitor daily, weekly, monthly portfolio returns. 
Use integrated gradients for attribution features in order to attribute input data based on model's predictions, in consideration of model interpretability. 
Deployed on Cloud.

## Objective
The project involves implementing and comparing multiple deep reinforcement learning algorithms (A2C, DDPG, PPO) for financial market trading. The goal is to understand how these algorithms work, compare their performance, and add explainability features to provide insights into and optimize trading decisions.

## Input Data:
**Key Implementations can be referenced in scripts/main.py**

Note: There are 29 unique tics (stocks) in the train dataset.

```python
stock_dimension = len(train.tic.unique()) # 29
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension # 291
```
## Env Parameter:
```python
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


e_train_gym = StockTradingEnv(df = train, **env_kwargs)
```
### Environment for Trading:
```python
env_train, _ = e_train_gym.get_sb_env()
```
