# TensorBoard Setup and Enhanced Metrics Guide

## Quick Start

### Launch TensorBoard (Two Options)

**Option 1: Python Script (Recommended)**
```bash
python3 launch_tensorboard.py
```

**Option 2: Bash Script (Fallback)**
```bash
./launch_tensorboard.sh
```

**View Specific Training Runs:**
```bash
# View all current and future runs
python3 launch_tensorboard.py

# View only legacy runs (6 separate runs organized by date & algorithm)
python3 launch_tensorboard.py --logdir tensorboard_log/legacy

# View specific legacy runs by algorithm
python3 launch_tensorboard.py --logdir tensorboard_log/legacy/run_3_ddpg_jul25_batch
python3 launch_tensorboard.py --logdir tensorboard_log/legacy/run_5_a2c_jul28_batch
python3 launch_tensorboard.py --logdir tensorboard_log/legacy/run_6_ppo_jul29

# View specific algorithm runs (when available for future runs)
python3 launch_tensorboard.py --logdir tensorboard_log/a2c
python3 launch_tensorboard.py --logdir tensorboard_log/ppo
python3 launch_tensorboard.py --logdir tensorboard_log/ddpg

# Custom port/host
python3 launch_tensorboard.py --port 6007 --host 0.0.0.0
```

## Enhanced Metrics Available

### 📊 Standard Training Metrics
- `train/reward` - Step-by-step rewards
- `train/reward_mean`, `train/reward_min`, `train/reward_max` - Reward statistics
- `train/reward_std` - Reward variance
- `train/learning_rate` - Current learning rate
- `train/entropy_loss`, `train/policy_loss`, `train/value_loss` - Algorithm-specific losses

### 📈 Portfolio-Specific Metrics (NEW)
- `portfolio/sharpe_ratio` - Risk-adjusted returns (annualized)
- `portfolio/max_drawdown` - Maximum portfolio decline
- `portfolio/total_return` - Cumulative return percentage
- `portfolio/current_value` - Real-time portfolio value
- `portfolio/episode_return` - Return per trading episode

### 🎯 Action Analysis Metrics (NEW)
- `actions/mean` - Average action values
- `actions/std` - Action variance/volatility
- `actions/sparsity` - Percentage of near-zero actions (strategy conservatism)

### 📺 Episode-Level Metrics (NEW)
- `episode/length` - Steps per trading episode
- `episode/reward` - Total reward per episode
- `episode/count` - Episode counter
- `episode/reward_mean` - Rolling average episode rewards
- `episode/length_mean` - Rolling average episode lengths

### 🔧 Training Diagnostics (NEW)
- `train/grad_norm` - Gradient magnitude (training stability indicator)

## File Organization

### Current Structure
```
tensorboard_log/
├── a2c/           # A2C algorithm logs (future runs)
├── ppo/           # PPO algorithm logs (future runs)
├── ddpg/          # DDPG algorithm logs (future runs)  
├── ensemble/      # Ensemble strategy logs (future runs)
└── legacy/        # Historical runs (organized by date & algorithm)
    ├── run_1_ppo_jul15/         # July 15 PPO training
    ├── run_2_ppo_jul20/         # July 20 PPO training  
    ├── run_3_ddpg_jul25_batch/  # July 25 DDPG batch (7 runs)
    ├── run_4_ddpg_jul26/        # July 26 DDPG training (2 runs)
    ├── run_5_a2c_jul28_batch/   # July 28 A2C batch (3 runs)
    └── run_6_ppo_jul29/         # July 29 PPO training (3 runs)
```

### Event Files
- **Old location**: `results/{algorithm}/events.out.tfevents.*` (moved to legacy/)
- **Current location**: `tensorboard_log/{algorithm}/{run_name}/events.out.tfevents.*`

## Process Management

### Automatic Cleanup
Both launch scripts automatically:
- ✅ **Kill existing TensorBoard processes** before starting
- ✅ **Clean up on exit** (Ctrl+C, script termination)
- ✅ **Handle signals gracefully** (SIGINT, SIGTERM)

### Manual Cleanup
If you need to manually kill TensorBoard processes:
```bash
# Using the dedicated cleanup script
python3 kill_tensorboard.py

# Or manually
pkill -f tensorboard
```

## Troubleshooting

### TensorBoard Won't Start
1. **Python Environment Issues**: Use the bash script instead
   ```bash
   ./launch_tensorboard.sh
   ```

2. **Port Already in Use**: Scripts automatically find available ports
   ```bash
   python3 launch_tensorboard.py --port 6007
   ```

3. **Processes Still Running**: Scripts automatically clean up, but you can force cleanup
   ```bash
   python3 kill_tensorboard.py
   ```

4. **No Data Visible**: Check log directory
   ```bash
   ls -la tensorboard_log/
   ```

### Missing Metrics
- Portfolio metrics require environment to provide `portfolio_value` in info dict
- Action metrics available for all algorithms
- Episode metrics track automatically
- Gradient norms require gradients (training phase only)

## Metric Interpretation

### Portfolio Metrics
- **Sharpe Ratio > 1.0**: Good risk-adjusted performance
- **Max Drawdown < 10%**: Conservative risk management
- **Total Return**: Compare against benchmark (e.g., S&P 500)

### Action Metrics
- **High Sparsity (>0.7)**: Conservative trading strategy
- **Low Sparsity (<0.3)**: Aggressive trading strategy
- **Action Std**: Measure of trading volatility

### Training Diagnostics
- **Gradient Norm**: Should decrease over time; spikes indicate instability
- **Reward Std**: Lower variance indicates more consistent learning

## Algorithm-Specific Considerations

### A2C/PPO (On-Policy)
- Episode metrics more meaningful
- Portfolio analysis from rollout buffer
- Entropy loss tracks exploration

### DDPG (Off-Policy)
- Replay buffer metrics
- Action noise critical for exploration
- Policy/value loss separation

## Integration with Existing Workflow

The enhanced metrics are automatically logged when you use:
```python
from agents.DRLAgent import DRLAgent

# Training automatically includes enhanced callback
model = DRLAgent.train_model(
    model=model,
    tb_log_name="enhanced_run",
    total_timesteps=100000
)
```

No code changes needed - metrics are logged transparently through the enhanced `TensorboardCallback`.