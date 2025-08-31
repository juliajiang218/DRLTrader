import numpy as np
import pandas as pd

def process_mvo_df(df, stock_dimension):
    """
    
    """
    df = df.sort_values(['date', 'tic'], ignore_index=True)[['date','tic','close']]
    fst = df
    fst = fst.iloc[0:stock_dimension, :]
    tic = fst['tic'].tolist()

    mvo = pd.DataFrame()

    for k in range(len(tic)):
        mvo[tic[k]] = 0
    
    for i in range(df.shape[0] // stock_dimension):
        n = df
        n = n.iloc[i * stock_dimension: (i+1) * stock_dimension, :]
        date= n['date'][i*stock_dimension]
        mvo.loc[date] = n['close'].tolist()

    return mvo

def StockReturnsComputing(stockPrice, rows, columns):
    """
    This calculates ?
    """
    import numpy as np 
    
    stockReturn = np.zeros([rows-1, columns])
    for j in range(columns): #j: assets
        for i in range(rows-1): #j: daily prices
            stockReturn[i,j] = ((stockPrice[i+1, j] - stockPrice[i, j])/stockPrice[i,j])*100

    return stockReturn

from pypfopt.efficient_frontier import EfficientFrontier
def compute_mvo_weights(meanReturns, covReturns, stock_dimension):
    ef_mean= EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean=ef_mean.clean_weights()
    mvo_weights=np.array([1000000 * cleaned_weights_mean[i] for i in range(stock_dimension)])

    return mvo_weights


import datetime
import json
import os
import numpy as np

def save_episode_report(model, environment, step_data, total_reward, step_count, 
                       model_name="Unknown", deterministic=True):
    """
    Save episode trace data to a comprehensive readable report.
    
    Args:
        model: The trained model used
        environment: Trading environment
        step_data: List of dictionaries containing step information
        total_reward: Final cumulative reward
        step_count: Total number of steps
        model_name: Name of the model
        deterministic: Whether deterministic prediction was used
    """
    
    def _extract_policy_details(model):
        """Extract comprehensive policy information from the model"""
        policy_details = {}
        
        try:
            policy = model.policy
            policy_details['policy_class'] = type(policy).__name__
            policy_details['policy_type'] = str(policy.__class__.__module__ + "." + policy.__class__.__name__)
            
            # Get action and observation space details
            if hasattr(policy, 'action_space'):
                action_space = policy.action_space
                policy_details['action_space_type'] = type(action_space).__name__
                policy_details['action_space_shape'] = getattr(action_space, 'shape', 'Unknown')
                policy_details['action_space_size'] = getattr(action_space, 'n', getattr(action_space, 'shape', 'Unknown'))
                if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                    policy_details['action_space_low'] = action_space.low.tolist() if hasattr(action_space.low, 'tolist') else str(action_space.low)
                    policy_details['action_space_high'] = action_space.high.tolist() if hasattr(action_space.high, 'tolist') else str(action_space.high)
            
            if hasattr(policy, 'observation_space'):
                obs_space = policy.observation_space
                policy_details['observation_space_type'] = type(obs_space).__name__
                policy_details['observation_space_shape'] = getattr(obs_space, 'shape', 'Unknown')
            
            # Get network architecture details
            if hasattr(policy, 'mlp_extractor'):
                mlp = policy.mlp_extractor
                policy_details['mlp_extractor_class'] = type(mlp).__name__
                
                # Get policy network details
                if hasattr(mlp, 'policy_net'):
                    policy_net = mlp.policy_net
                    policy_details['policy_network_layers'] = []
                    for i, layer in enumerate(policy_net):
                        layer_info = {
                            'layer_index': i,
                            'layer_type': type(layer).__name__,
                            'layer_params': {}
                        }
                        
                        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                            layer_info['layer_params']['in_features'] = layer.in_features
                            layer_info['layer_params']['out_features'] = layer.out_features
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            layer_info['layer_params']['has_bias'] = True
                        
                        policy_details['policy_network_layers'].append(layer_info)
                
                # Get value network details  
                if hasattr(mlp, 'value_net'):
                    value_net = mlp.value_net
                    policy_details['value_network_layers'] = []
                    for i, layer in enumerate(value_net):
                        layer_info = {
                            'layer_index': i,
                            'layer_type': type(layer).__name__,
                            'layer_params': {}
                        }
                        
                        if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                            layer_info['layer_params']['in_features'] = layer.in_features
                            layer_info['layer_params']['out_features'] = layer.out_features
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            layer_info['layer_params']['has_bias'] = True
                            
                        policy_details['value_network_layers'].append(layer_info)
            
            # Get action distribution details
            if hasattr(policy, 'action_dist'):
                action_dist = policy.action_dist
                policy_details['action_distribution_class'] = type(action_dist).__name__
            
            # Get optimizer details if available
            if hasattr(model, 'policy') and hasattr(model.policy, 'optimizer'):
                optimizer = model.policy.optimizer
                policy_details['optimizer_class'] = type(optimizer).__name__
                policy_details['optimizer_params'] = {}
                for param_group in optimizer.param_groups:
                    for key, value in param_group.items():
                        if key != 'params':  # Skip the actual parameters
                            policy_details['optimizer_params'][key] = value
            
            # Get learning rate
            if hasattr(model, 'learning_rate'):
                policy_details['learning_rate'] = float(model.learning_rate)
            
            # Get policy kwargs if available
            if hasattr(model, 'policy_kwargs'):
                policy_details['policy_kwargs'] = model.policy_kwargs
                
            # Count total parameters
            total_params = 0
            trainable_params = 0
            if hasattr(policy, 'parameters'):
                for param in policy.parameters():
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                        
            policy_details['total_parameters'] = total_params
            policy_details['trainable_parameters'] = trainable_params
            
        except Exception as e:
            policy_details['extraction_error'] = str(e)
            
        return policy_details
    
    # Import training dates from main.py
    try:
        from main import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
    except ImportError as e:
        # Report import error and use fallback values
        TRAIN_START_DATE = 'Import Error'
        TRAIN_END_DATE = 'Import Error'
        TEST_START_DATE = 'Import Error'
        TEST_END_DATE = 'Import Error'
        import_error = f"Failed to import training dates from main.py: {str(e)}"
    else:
        import_error = None
    
    # Calculate training duration
    try:
        train_start = datetime.datetime.strptime(TRAIN_START_DATE, '%Y-%m-%d')
        train_end = datetime.datetime.strptime(TRAIN_END_DATE, '%Y-%m-%d')
        training_duration_days = (train_end - train_start).days
        training_duration_years = training_duration_days / 365.25
        duration_error = None
    except Exception as e:
        training_duration_days = "Calculation Error"
        training_duration_years = "Calculation Error"
        duration_error = f"Failed to calculate training duration: {str(e)}"
    
    # Get model timesteps and convert to scientific notation
    model_timesteps = getattr(model, 'num_timesteps', 0)
    
    # Convert timesteps to 1e6 format
    if isinstance(model_timesteps, (int, float)) and model_timesteps != 0:
        if model_timesteps >= 1e6:
            timesteps_formatted = f"{model_timesteps:.0e}".replace('+0', '').replace('+', '')
        else:
            timesteps_formatted = str(int(model_timesteps))
    else:
        timesteps_formatted = 'Unknown'
    
    # Create shorter timestamp (YYMMDD_HHMM)
    short_timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    
    # Create report filename (will overwrite if same name+timesteps)
    report_filename = f"Episode_Report_{model_name}_{timesteps_formatted}.txt"
    
    # Ensure reports directory exists
    reports_dir = "episode_reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, report_filename)
    
    # Gather model information
    model_info = {
        'model_name': model_name,
        'model_class': type(model).__name__,
        'num_timesteps': model_timesteps,
        'deterministic_mode': deterministic
    }
    
    # Extract policy details
    policy_info = _extract_policy_details(model)
    
    # Gather training period information
    training_info = {
        'train_start_date': TRAIN_START_DATE,
        'train_end_date': TRAIN_END_DATE,
        'training_duration_days': training_duration_days,
        'training_duration_years': f"{training_duration_years:.2f}" if isinstance(training_duration_years, (int, float)) else training_duration_years,
        'test_start_date': TEST_START_DATE,
        'test_end_date': TEST_END_DATE
    }
    
    # Add error information if present
    if import_error:
        training_info['import_error'] = import_error
    if duration_error:
        training_info['duration_calculation_error'] = duration_error
    
    # Gather environment information
    env_info = {
        'initial_account_value': getattr(environment, 'initial_total_asset', 'Unknown'),
        'trading_days': len(environment.df.index.unique()) if hasattr(environment, 'df') else 'Unknown',
        'stock_dimension': getattr(environment, 'stock_dim', 'Unknown'),
        'action_space_size': environment.action_space.shape if hasattr(environment, 'action_space') else 'Unknown'
    }
    
    # Create comprehensive report
    with open(report_path, 'w') as f:
        # Header with short timestamp
        f.write("=" * 80 + "\n")
        f.write(f"EPISODE TRACE REPORT - {short_timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Model Information Section
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in model_info.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Policy Information Section
        f.write("POLICY DETAILS:\n")
        f.write("-" * 40 + "\n")
        
        # Basic policy info
        basic_policy_keys = ['policy_class', 'policy_type', 'action_distribution_class', 
                           'learning_rate', 'total_parameters', 'trainable_parameters']
        for key in basic_policy_keys:
            if key in policy_info:
                f.write(f"{key.replace('_', ' ').title()}: {policy_info[key]}\n")
        
        # Action space details
        if 'action_space_type' in policy_info:
            f.write(f"\nAction Space:\n")
            f.write(f"  Type: {policy_info['action_space_type']}\n")
            f.write(f"  Shape: {policy_info.get('action_space_shape', 'Unknown')}\n")
            f.write(f"  Size: {policy_info.get('action_space_size', 'Unknown')}\n")
            if 'action_space_low' in policy_info:
                f.write(f"  Low: {policy_info['action_space_low']}\n")
            if 'action_space_high' in policy_info:
                f.write(f"  High: {policy_info['action_space_high']}\n")
        
        # Observation space details
        if 'observation_space_type' in policy_info:
            f.write(f"\nObservation Space:\n")
            f.write(f"  Type: {policy_info['observation_space_type']}\n")
            f.write(f"  Shape: {policy_info.get('observation_space_shape', 'Unknown')}\n")
        
        # Network architecture
        if 'policy_network_layers' in policy_info:
            f.write(f"\nPolicy Network Architecture:\n")
            for layer in policy_info['policy_network_layers']:
                f.write(f"  Layer {layer['layer_index']}: {layer['layer_type']}")
                if 'in_features' in layer['layer_params'] and 'out_features' in layer['layer_params']:
                    f.write(f" ({layer['layer_params']['in_features']} -> {layer['layer_params']['out_features']})")
                f.write("\n")
        
        if 'value_network_layers' in policy_info:
            f.write(f"\nValue Network Architecture:\n")
            for layer in policy_info['value_network_layers']:
                f.write(f"  Layer {layer['layer_index']}: {layer['layer_type']}")
                if 'in_features' in layer['layer_params'] and 'out_features' in layer['layer_params']:
                    f.write(f" ({layer['layer_params']['in_features']} -> {layer['layer_params']['out_features']})")
                f.write("\n")
        
        # Optimizer details
        if 'optimizer_class' in policy_info:
            f.write(f"\nOptimizer Details:\n")
            f.write(f"  Class: {policy_info['optimizer_class']}\n")
            if 'optimizer_params' in policy_info:
                for key, value in policy_info['optimizer_params'].items():
                    f.write(f"  {key.title()}: {value}\n")
        
        # Policy kwargs
        if 'policy_kwargs' in policy_info and policy_info['policy_kwargs']:
            f.write(f"\nPolicy Configuration:\n")
            for key, value in policy_info['policy_kwargs'].items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\n")
        
        # Training Period Information Section
        f.write("TRAINING PERIOD INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in training_info.items():
            if 'error' in key.lower():
                f.write(f"ERROR - {key.replace('_', ' ').title()}: {value}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Environment Information Section
        f.write("ENVIRONMENT INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in env_info.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Episode Summary
        f.write("EPISODE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Steps: {step_count}\n")
        f.write(f"Total Reward: {total_reward:.6f}\n")
        f.write(f"Average Reward per Step: {total_reward/step_count if step_count > 0 else 0:.6f}\n")
        f.write(f"Episode Completion: {'Full Episode' if step_count >= len(environment.df.index.unique())-1 else 'Early Termination'}\n")
        f.write("\n")
        
        # Detailed Step-by-Step Information
        f.write("DETAILED STEP TRACE:\n")
        f.write("-" * 40 + "\n")
        
        for i, step in enumerate(step_data):
            f.write(f"Step {i}:\n")
            f.write(f"  Date/Day: {step.get('day', 'Unknown')}\n")
            f.write(f"  Action: {step.get('action', 'Unknown')}\n")
            f.write(f"  Reward: {step.get('reward', 'Unknown')}\n")
            f.write(f"  Account Value: {step.get('account_value', 'Unknown')}\n")
            f.write(f"  Done: {step.get('done', 'Unknown')}\n")
            
            # Add policy information if available
            if 'policy_mean' in step:
                f.write(f"  Policy Mean: {step['policy_mean']}\n")
            if 'policy_std' in step:
                f.write(f"  Policy Std: {step['policy_std']}\n")
                
            f.write("\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        
        # Calculate additional metrics if data available
        rewards = [step.get('reward', 0) for step in step_data if step.get('reward') is not None]
        if rewards:
            f.write(f"Max Single Step Reward: {max(rewards):.6f}\n")
            f.write(f"Min Single Step Reward: {min(rewards):.6f}\n")
            f.write(f"Reward Standard Deviation: {np.std(rewards):.6f}\n")
        
        account_values = [step.get('account_value') for step in step_data if step.get('account_value') is not None]
        if account_values and len(account_values) > 1:
            initial_value = account_values[0]
            final_value = account_values[-1]
            f.write(f"Portfolio Return: {((final_value/initial_value - 1) * 100):.2f}%\n")
        
        f.write("\n")
        
        # Training vs Testing Period Summary (only if no errors)
        if not import_error:
            f.write("PERIOD SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model was trained on data from {TRAIN_START_DATE} to {TRAIN_END_DATE}\n")
            if isinstance(training_duration_years, (int, float)):
                f.write(f"Training period duration: {training_duration_days} days ({training_duration_years:.2f} years)\n")
            f.write(f"Episode tested on data from {TEST_START_DATE} to {TEST_END_DATE}\n")
            f.write("\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n📊 Episode report saved to: {report_path}")
    
    # Also save as JSON for programmatic access
    json_filename = f"Episode_Report_{model_name}_{timesteps_formatted}.json"
    json_path = os.path.join(reports_dir, json_filename)
    
    report_data = {
        'timestamp': short_timestamp,
        'model_info': model_info,
        'policy_info': policy_info,
        'training_info': training_info,
        'environment_info': env_info,
        'episode_summary': {
            'total_steps': step_count,
            'total_reward': total_reward,
            'average_reward_per_step': total_reward/step_count if step_count > 0 else 0
        },
        'step_data': step_data,
        'performance_metrics': {
            'max_reward': max(rewards) if rewards else None,
            'min_reward': min(rewards) if rewards else None,
            'reward_std': float(np.std(rewards)) if rewards else None,
            'portfolio_return': ((account_values[-1]/account_values[0] - 1) * 100) if len(account_values) > 1 else None
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"📊 JSON report saved to: {json_path}")
    
    return report_path, json_path