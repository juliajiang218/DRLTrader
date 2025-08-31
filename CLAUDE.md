# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Reinforcement Learning (DRL) ensemble trading system that trains and compares multiple RL algorithms (A2C, PPO, DDPG) for financial market trading. The system uses an ensemble strategy that selects the best-performing model based on Sharpe ratios for each trading period.

## Key Architecture Components

### Core Components
- **Trading Environment**: `EnvTrade/env_trading.py` - Custom OpenAI Gym environment for stock trading simulation
- **DRL Agents**: `agents/DRLAgent.py` - Individual RL algorithm implementations
- **Ensemble Agent**: `agents/DRLEnsembleAgent.py` - Ensemble strategy combining multiple algorithms
- **Data Processing**: `preprocessor/yahoodownloader.py` - Yahoo Finance data fetching and preprocessing

### Environment Parameters
- **Stock Universe**: 29 unique stocks (DOW_30_TICKER subset)
- **State Space**: 291 dimensions (1 + 2*stock_dim + indicators*stock_dim)
- **Action Space**: 29 dimensions (one per stock, continuous values for position sizing)
- **Initial Capital**: $1,000,000
- **Transaction Costs**: 0.1% for both buying and selling
- **Max Holdings**: 100 shares per stock (hmax parameter)

### Ensemble Strategy
The ensemble uses a rolling window approach:
- **Rebalance Window**: 63 days (quarterly rebalancing)
- **Validation Window**: 63 days for model selection
- **Model Selection**: Based on Sharpe ratio performance during validation period
- **Turbulence Threshold**: Dynamic threshold based on market volatility (90th percentile)

## Common Commands

### Training Individual Models
```bash
# Train individual models (uncomment specific lines in main.py)
cd scripts/
python3 main.py
```

### Training Ensemble Strategy
```bash
# Run ensemble training (default behavior)
cd scripts/
python3 main.py
```

### Model Evaluation
```bash
# Evaluate trained models on test data
cd scripts/
python3 evaluate.py
```

### SLURM Cluster Execution
```bash
# Submit to SLURM cluster
cd slurm/
sbatch main.slurm

# Monitor jobs
squeue -u $USER
```

### Dependencies Installation
```bash
# Install required packages
pip3 install -r scripts/requirement.txt

# Or install individually:
pip3 install pandas numpy matplotlib seaborn yfinance requests gymnasium stable-baselines3 finrl
```

## Data Pipeline

### Input Data Structure
- **Training Data**: `datasets/train_data.csv` (2009-01-02 to 2020-06-30)
- **Testing Data**: `datasets/trade_data.csv` (2020-07-01 to 2021-10-27)
- **Required Columns**: date, tic (ticker), close, plus technical indicators

### Technical Indicators
The system uses multiple technical indicators defined in `finrl.config.INDICATORS`:
- Moving averages, RSI, MACD, Bollinger Bands, etc.
- Each indicator adds dimensions to the state space

## Model Storage and Outputs

### Trained Models Location
- **Individual Models**: `trained_models/AGENT_{algorithm}.zip`
- **Ensemble Models**: `trained_models/{ALGORITHM}_{timesteps}k_{iteration}.zip`

### Results Structure
- **Account Values**: `results/account_value_{validation/trade}_{model}_{iteration}.csv`
- **Actions**: `results/actions_{validation/trade}_{model}_{iteration}.csv`
- **Rewards**: `results/account_rewards_{validation/trade}_{model}_{iteration}.csv`
- **Training Logs**: `results/{algorithm}/progress.csv`
- **TensorBoard Logs**: `tensorboard_log/{algorithm}/`

### Visualizations
- **Training Metrics**: `results/graphs/{algorithm}_training_metrics.png`
- **Backtest Results**: `results/graphs/backtest_result.png`

## Key Configuration Notes

### Model Hyperparameters
- **A2C**: n_steps=5, ent_coef=0.005, learning_rate=0.0007
- **PPO**: n_steps=2048, ent_coef=0.001, learning_rate=0.0001, batch_size=256
- **DDPG**: buffer_size=10,000, learning_rate=0.0005, batch_size=128

### Training Configuration
- **Default Timesteps**: 10,000,000 for individual models
- **Ensemble Timesteps**: 100 per model (for faster iteration)
- **Reward Scaling**: 1e-4

## Development Notes

### Data Dependencies
- Requires FinRL library and custom environment setup
- Yahoo Finance data fetching through custom downloader
- Parquet files in `datasets/` contain preprocessed training data

### SLURM Environment
- Configured for Wake Forest DEAC cluster
- Memory allocation: 102GB
- Time limit: 2 days
- Uses account: csc790

### Episode Reporting
- Detailed episode reports saved to `episode_reports/`
- JSON and text formats available
- Includes step-by-step trading decisions and portfolio evolution