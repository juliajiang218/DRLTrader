import datetime
import json
import os
import numpy as np


class ModelInfoExtractor:
    """Extracts basic model information."""
    
    def extract(self, model, model_name="Unknown", deterministic=True):
        """Extract basic model information."""
        model_timesteps = getattr(model, 'num_timesteps', 0)
        
        # Convert timesteps to 1e6 format
        if isinstance(model_timesteps, (int, float)) and model_timesteps != 0:
            if model_timesteps >= 1e6:
                timesteps_formatted = f"{model_timesteps:.0e}".replace('+0', '').replace('+', '')
            else:
                timesteps_formatted = str(int(model_timesteps))
        else:
            timesteps_formatted = 'Unknown'
        
        return {
            'model_name': model_name,
            'model_class': type(model).__name__,
            'timesteps': model_timesteps,
            'timesteps_formatted': timesteps_formatted,
            'deterministic_prediction': deterministic
        }


class PolicyInfoExtractor:
    """Extracts comprehensive policy information from the model."""
    
    def extract(self, model):
        """Extract comprehensive policy information from the model."""
        policy_details = {}
        
        try:
            policy = model.policy
            policy_details['policy_class'] = type(policy).__name__
            policy_details['policy_type'] = str(policy.__class__.__module__ + "." + policy.__class__.__name__)
            
            # Get network architecture details
            if hasattr(policy, 'mlp_extractor'):
                policy_details['has_mlp_extractor'] = True
                mlp = policy.mlp_extractor
                policy_details['mlp_extractor_type'] = type(mlp).__name__
                
                # Extract MLP extractor architecture details
                if hasattr(mlp, 'policy_net'):
                    policy_details['policy_network_layers'] = self._extract_network_structure(mlp.policy_net)
                if hasattr(mlp, 'value_net'):
                    policy_details['value_network_layers'] = self._extract_network_structure(mlp.value_net)
                
                # Get shared layers information
                if hasattr(mlp, 'shared_net'):
                    policy_details['shared_network_layers'] = self._extract_network_structure(mlp.shared_net)
            
            # Action and Value network details
            if hasattr(policy, 'action_net'):
                policy_details['action_net_type'] = type(policy.action_net).__name__
                policy_details['action_net_structure'] = self._extract_network_structure(policy.action_net)
                
                # For continuous action spaces, get distribution info
                if hasattr(policy, 'action_dist'):
                    policy_details['action_distribution_type'] = type(policy.action_dist).__name__
            
            if hasattr(policy, 'value_net'):
                policy_details['value_net_type'] = type(policy.value_net).__name__
                policy_details['value_net_structure'] = self._extract_network_structure(policy.value_net)
            
            # Features extractor details
            if hasattr(policy, 'features_extractor'):
                features_extractor = policy.features_extractor
                policy_details['features_extractor_type'] = type(features_extractor).__name__
                policy_details['features_extractor_structure'] = self._extract_network_structure(features_extractor)
            
            # Get activation function information
            if hasattr(policy, 'activation_fn'):
                policy_details['activation_function'] = str(policy.activation_fn)
            
            # Get network architecture (net_arch)
            if hasattr(policy, 'net_arch'):
                policy_details['network_architecture'] = policy.net_arch
            
            # Algorithm-specific network details
            model_class = type(model).__name__.lower()
            if 'ddpg' in model_class or 'td3' in model_class:
                # For DDPG/TD3: Actor-Critic architecture
                if hasattr(policy, 'actor'):
                    policy_details['actor_network_structure'] = self._extract_network_structure(policy.actor)
                if hasattr(policy, 'critic'):
                    policy_details['critic_network_structure'] = self._extract_network_structure(policy.critic)
                if hasattr(policy, 'critic_target'):
                    policy_details['has_target_networks'] = True
            
            # Count parameters by network component
            self._count_parameters_by_component(policy, policy_details)
            
            # Count total parameters
            total_params = 0
            trainable_params = 0
            
            for param in policy.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
            policy_details['total_parameters'] = total_params
            policy_details['trainable_parameters'] = trainable_params
            
        except Exception as e:
            policy_details['extraction_error'] = str(e)
            
        return policy_details
    
    def _extract_network_structure(self, network):
        """Extract the structure of a neural network."""
        if network is None:
            return "None"
        
        try:
            structure = []
            if hasattr(network, 'children'):
                for i, layer in enumerate(network.children()):
                    layer_info = {
                        'layer_index': i,
                        'layer_type': type(layer).__name__,
                    }
                    
                    # Get layer-specific details
                    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                        layer_info['input_features'] = layer.in_features
                        layer_info['output_features'] = layer.out_features
                    
                    if hasattr(layer, 'kernel_size'):
                        layer_info['kernel_size'] = layer.kernel_size
                    
                    if hasattr(layer, 'stride'):
                        layer_info['stride'] = layer.stride
                        
                    if hasattr(layer, 'padding'):
                        layer_info['padding'] = layer.padding
                    
                    structure.append(layer_info)
            
            return structure if structure else str(network)
            
        except Exception as e:
            return f"Structure extraction error: {str(e)}"
    
    def _count_parameters_by_component(self, policy, policy_details):
        """Count parameters for each network component."""
        try:
            # Count parameters in different components
            if hasattr(policy, 'mlp_extractor'):
                mlp = policy.mlp_extractor
                
                if hasattr(mlp, 'policy_net'):
                    policy_net_params = sum(p.numel() for p in mlp.policy_net.parameters())
                    policy_details['policy_network_parameters'] = policy_net_params
                
                if hasattr(mlp, 'value_net'):
                    value_net_params = sum(p.numel() for p in mlp.value_net.parameters())
                    policy_details['value_network_parameters'] = value_net_params
                
                if hasattr(mlp, 'shared_net'):
                    shared_net_params = sum(p.numel() for p in mlp.shared_net.parameters())
                    policy_details['shared_network_parameters'] = shared_net_params
            
            if hasattr(policy, 'action_net'):
                action_net_params = sum(p.numel() for p in policy.action_net.parameters())
                policy_details['action_net_parameters'] = action_net_params
            
            if hasattr(policy, 'value_net'):
                value_net_params = sum(p.numel() for p in policy.value_net.parameters())
                policy_details['value_net_parameters'] = value_net_params
            
            if hasattr(policy, 'features_extractor'):
                features_params = sum(p.numel() for p in policy.features_extractor.parameters())
                policy_details['features_extractor_parameters'] = features_params
                
        except Exception as e:
            policy_details['parameter_counting_error'] = str(e)


class HyperparameterExtractor:
    """Extracts hyper-parameters from the trained model."""
    
    def extract(self, model):
        """Extract hyper-parameters from the trained model."""
        hyperparams = {}
        
        try:
            # Get basic model attributes that are typically hyper-parameters
            hyperparam_attrs = [
                'learning_rate', 'lr', 'gamma', 'gae_lambda', 'clip_range', 
                'clip_range_vf', 'ent_coef', 'vf_coef', 'max_grad_norm',
                'n_steps', 'batch_size', 'n_epochs', 'buffer_size',
                'train_freq', 'gradient_steps', 'target_update_interval',
                'tau', 'policy_delay', 'noise_clip', 'action_noise',
                'exploration_fraction', 'exploration_final_eps',
                'exploration_initial_eps', 'tensorboard_log', 'verbose'
            ]
            
            for attr in hyperparam_attrs:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    hyperparams[attr] = str(value) if not isinstance(value, (int, float, bool, str, type(None))) else value
            
            # Extract optimizer parameters if available
            if hasattr(model, 'policy') and hasattr(model.policy, 'optimizer'):
                optimizer = model.policy.optimizer
                hyperparams['optimizer_type'] = type(optimizer).__name__
                if hasattr(optimizer, 'param_groups'):
                    for i, group in enumerate(optimizer.param_groups):
                        for key, value in group.items():
                            if key != 'params':  # Skip the actual parameters
                                hyperparams[f'optimizer_group_{i}_{key}'] = value
            
            # Extract network architecture parameters
            if hasattr(model, 'policy'):
                policy = model.policy
                if hasattr(policy, 'net_arch'):
                    hyperparams['net_arch'] = policy.net_arch
                if hasattr(policy, 'activation_fn'):
                    hyperparams['activation_fn'] = str(policy.activation_fn)
                if hasattr(policy, 'features_extractor'):
                    hyperparams['features_extractor_class'] = type(policy.features_extractor).__name__
            
            # Extract algorithm-specific parameters
            model_class = type(model).__name__.lower()
            
            if 'ppo' in model_class:
                ppo_attrs = ['clip_range', 'clip_range_vf', 'normalize_advantage', 'use_sde']
                for attr in ppo_attrs:
                    if hasattr(model, attr):
                        hyperparams[attr] = getattr(model, attr)
            
            elif 'a2c' in model_class:
                a2c_attrs = ['use_rms_prop', 'rms_prop_eps', 'normalize_advantage', 'use_sde']
                for attr in a2c_attrs:
                    if hasattr(model, attr):
                        hyperparams[attr] = getattr(model, attr)
            
            elif 'ddpg' in model_class or 'td3' in model_class:
                ddpg_attrs = ['action_noise', 'policy_delay', 'noise_clip']
                for attr in ddpg_attrs:
                    if hasattr(model, attr):
                        hyperparams[attr] = str(getattr(model, attr))
            
            # Get training timesteps
            if hasattr(model, 'num_timesteps'):
                hyperparams['num_timesteps'] = model.num_timesteps
            
            # Get device information
            if hasattr(model, 'device'):
                hyperparams['device'] = str(model.device)
            
        except Exception as e:
            hyperparams['extraction_error'] = str(e)
        
        return hyperparams


class TrainingInfoExtractor:
    """Extracts training period information."""
    
    def extract(self):
        """Extract training period information from scripts.main following directory structure."""
        TRAIN_START_DATE = 'Not Available'
        TRAIN_END_DATE = 'Not Available'
        TEST_START_DATE = 'Not Available'
        TEST_END_DATE = 'Not Available'
        import_error = None
        
        try:
            # Follow the correct directory structure
            # From EpisodeReporter/reporter.py -> go up one level to project root -> then to scripts/main.py
            import sys
            import os
            
            # Add the project root directory to Python path (go up one directory from EpisodeReporter/)
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Import from scripts.main as specified
            from scripts.main import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
            
        except ImportError as e:
            # Fallback: Try alternative import methods
            try:
                # Method 2: Try direct import from main
                from main import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
            except ImportError:
                # Method 3: Use reasonable defaults for demo/testing
                TRAIN_START_DATE = '2020-01-01'
                TRAIN_END_DATE = '2023-01-01'
                TEST_START_DATE = '2023-01-01'
                TEST_END_DATE = '2024-01-01'
                import_error = f"Training dates not found in scripts.main - using default values. Original error: {str(e)}"
        except Exception as e:
            # Handle any other errors
            TRAIN_START_DATE = '2020-01-01'
            TRAIN_END_DATE = '2023-01-01'
            TEST_START_DATE = '2023-01-01'
            TEST_END_DATE = '2024-01-01'
            import_error = f"Error importing training dates: {str(e)} - using default values"
        
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
        
        training_info = {
            'train_start_date': TRAIN_START_DATE,
            'train_end_date': TRAIN_END_DATE,
            'test_start_date': TEST_START_DATE,
            'test_end_date': TEST_END_DATE,
            'training_duration_days': training_duration_days,
            'training_duration_years': training_duration_years
        }
        
        if import_error:
            training_info['import_error'] = import_error
        if duration_error:
            training_info['duration_error'] = duration_error
            
        return training_info
        
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
        
        training_info = {
            'train_start_date': TRAIN_START_DATE,
            'train_end_date': TRAIN_END_DATE,
            'test_start_date': TEST_START_DATE,
            'test_end_date': TEST_END_DATE,
            'training_duration_days': training_duration_days,
            'training_duration_years': training_duration_years
        }
        
        if import_error:
            training_info['import_error'] = import_error
        if duration_error:
            training_info['duration_error'] = duration_error
            
        return training_info
        
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
        
        training_info = {
            'train_start_date': TRAIN_START_DATE,
            'train_end_date': TRAIN_END_DATE,
            'test_start_date': TEST_START_DATE,
            'test_end_date': TEST_END_DATE,
            'training_duration_days': training_duration_days,
            'training_duration_years': training_duration_years
        }
        
        if import_error:
            training_info['import_error'] = import_error
        if duration_error:
            training_info['duration_error'] = duration_error
            
        return training_info


class EnvironmentInfoExtractor:
    """Extracts environment information."""
    
    def extract(self, environment):
        """Extract environment information."""
        return {
            'environment_class': type(environment).__name__,
            'data_length': len(environment.df.index.unique()) if hasattr(environment, 'df') else 'Unknown',
            'initial_amount': getattr(environment, 'initial_amount', 'Unknown'),
            'stock_dimension': getattr(environment, 'stock_dim', 'Unknown'),
            'action_space': getattr(environment, 'action_space', 'Unknown'),
            'state_space': getattr(environment, 'state_space', 'Unknown')
        }


class PerformanceMetricsCalculator:
    """Calculates performance metrics from step data."""
    
    def calculate(self, step_data, total_reward, step_count, environment):
        """Calculate comprehensive performance metrics."""
        metrics = {
            'total_steps': step_count,
            'total_reward': total_reward,
            'average_reward_per_step': total_reward/step_count if step_count > 0 else 0,
            'episode_completion': 'Full Episode' if step_count >= len(environment.df.index.unique())-1 else 'Early Termination'
        }
        
        # Calculate reward statistics
        rewards = [step.get('reward', 0) for step in step_data if step.get('reward') is not None]
        if rewards:
            metrics.update({
                'max_reward': max(rewards),
                'min_reward': min(rewards),
                'reward_std': float(np.std(rewards))
            })
        
        # Calculate portfolio performance
        account_values = [step.get('account_value') for step in step_data if step.get('account_value') is not None]
        if account_values and len(account_values) > 1:
            initial_value = account_values[0]
            final_value = account_values[-1]
            metrics['portfolio_return'] = ((final_value/initial_value - 1) * 100)
        
        return metrics


class TextReportFormatter:
    """Handles text report formatting and saving."""
    
    def save_report(self, report_data, output_dir, filename):
        """Save comprehensive text report."""
        report_path = os.path.join(output_dir, filename)
        
        with open(report_path, 'w') as f:
            self._write_header(f, report_data)
            self._write_model_info(f, report_data['model_info'])
            self._write_policy_info(f, report_data['policy_info'])
            self._write_hyperparameters(f, report_data['hyperparameters'])
            self._write_training_info(f, report_data['training_info'])
            self._write_environment_info(f, report_data['environment_info'])
            self._write_episode_summary(f, report_data['performance_metrics'])
            self._write_step_details(f, report_data['episode_data']['step_data'])
            self._write_performance_metrics(f, report_data['performance_metrics'])
            self._write_period_summary(f, report_data['training_info'])
            self._write_footer(f)
        
        return report_path
    
    def _write_header(self, f, report_data):
        """Write report header."""
        short_timestamp = report_data['timestamp'].strftime("%y%m%d_%H%M")
        f.write("=" * 80 + "\n")
        f.write("DEEP REINFORCEMENT LEARNING TRADING EPISODE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {report_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Report ID: {short_timestamp}\n")
        f.write("\n")
    
    def _write_model_info(self, f, model_info):
        """Write model information section."""
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in model_info.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
    
    def _write_policy_info(self, f, policy_info):
        """Write policy information section with detailed network architecture."""
        f.write("POLICY INFORMATION:\n")
        f.write("-" * 40 + "\n")
        
        # Basic policy information
        basic_keys = ['policy_class', 'policy_type', 'has_mlp_extractor', 'mlp_extractor_type']
        for key in basic_keys:
            if key in policy_info:
                if 'error' in key.lower():
                    f.write(f"ERROR - {key.replace('_', ' ').title()}: {policy_info[key]}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {policy_info[key]}\n")
        
        # Network Architecture Details
        if 'network_architecture' in policy_info:
            f.write(f"Network Architecture: {policy_info['network_architecture']}\n")
        
        if 'activation_function' in policy_info:
            f.write(f"Activation Function: {policy_info['activation_function']}\n")
        
        # Features Extractor Details
        if 'features_extractor_type' in policy_info:
            f.write(f"\nFeatures Extractor:\n")
            f.write(f"  Type: {policy_info['features_extractor_type']}\n")
            if 'features_extractor_parameters' in policy_info:
                f.write(f"  Parameters: {policy_info['features_extractor_parameters']:,}\n")
            if 'features_extractor_structure' in policy_info:
                self._write_network_structure(f, "  Structure", policy_info['features_extractor_structure'])
        
        # Policy Network Details
        if 'policy_network_layers' in policy_info or 'policy_network_parameters' in policy_info:
            f.write(f"\nPolicy Network:\n")
            if 'policy_network_parameters' in policy_info:
                f.write(f"  Parameters: {policy_info['policy_network_parameters']:,}\n")
            if 'policy_network_layers' in policy_info:
                self._write_network_structure(f, "  Layers", policy_info['policy_network_layers'])
        
        # Value Network Details  
        if 'value_network_layers' in policy_info or 'value_network_parameters' in policy_info:
            f.write(f"\nValue Network:\n")
            if 'value_network_parameters' in policy_info:
                f.write(f"  Parameters: {policy_info['value_network_parameters']:,}\n")
            if 'value_network_layers' in policy_info:
                self._write_network_structure(f, "  Layers", policy_info['value_network_layers'])
        
        # Shared Network Details
        if 'shared_network_layers' in policy_info or 'shared_network_parameters' in policy_info:
            f.write(f"\nShared Network:\n")
            if 'shared_network_parameters' in policy_info:
                f.write(f"  Parameters: {policy_info['shared_network_parameters']:,}\n")
            if 'shared_network_layers' in policy_info:
                self._write_network_structure(f, "  Layers", policy_info['shared_network_layers'])
        
        # Action Network Details
        if 'action_net_type' in policy_info:
            f.write(f"\nAction Network:\n")
            f.write(f"  Type: {policy_info['action_net_type']}\n")
            if 'action_net_parameters' in policy_info:
                f.write(f"  Parameters: {policy_info['action_net_parameters']:,}\n")
            if 'action_net_structure' in policy_info:
                self._write_network_structure(f, "  Structure", policy_info['action_net_structure'])
            if 'action_distribution_type' in policy_info:
                f.write(f"  Action Distribution: {policy_info['action_distribution_type']}\n")
        
        # Value Network Details (separate from MLP value network)
        if 'value_net_type' in policy_info:
            f.write(f"\nValue Network (Output):\n")
            f.write(f"  Type: {policy_info['value_net_type']}\n")
            if 'value_net_parameters' in policy_info:
                f.write(f"  Parameters: {policy_info['value_net_parameters']:,}\n")
            if 'value_net_structure' in policy_info:
                self._write_network_structure(f, "  Structure", policy_info['value_net_structure'])
        
        # DDPG/TD3 specific networks
        if 'actor_network_structure' in policy_info:
            f.write(f"\nActor Network (DDPG/TD3):\n")
            self._write_network_structure(f, "  Structure", policy_info['actor_network_structure'])
        
        if 'critic_network_structure' in policy_info:
            f.write(f"\nCritic Network (DDPG/TD3):\n")
            self._write_network_structure(f, "  Structure", policy_info['critic_network_structure'])
        
        if 'has_target_networks' in policy_info:
            f.write(f"\nTarget Networks: {policy_info['has_target_networks']}\n")
        
        # Parameter Summary
        f.write(f"\nParameter Summary:\n")
        if 'total_parameters' in policy_info:
            f.write(f"  Total Parameters: {policy_info['total_parameters']:,}\n")
        if 'trainable_parameters' in policy_info:
            f.write(f"  Trainable Parameters: {policy_info['trainable_parameters']:,}\n")
        
        # Error reporting
        for key, value in policy_info.items():
            if 'error' in key.lower():
                f.write(f"ERROR - {key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\n")
    
    def _write_network_structure(self, f, title, structure):
        """Write network structure details."""
        if isinstance(structure, list) and structure:
            f.write(f"{title}:\n")
            for layer in structure:
                if isinstance(layer, dict):
                    layer_desc = f"    Layer {layer.get('layer_index', '?')}: {layer.get('layer_type', 'Unknown')}"
                    
                    if 'input_features' in layer and 'output_features' in layer:
                        layer_desc += f" ({layer['input_features']} -> {layer['output_features']})"
                    elif 'kernel_size' in layer:
                        layer_desc += f" (kernel: {layer['kernel_size']}"
                        if 'stride' in layer:
                            layer_desc += f", stride: {layer['stride']}"
                        if 'padding' in layer:
                            layer_desc += f", padding: {layer['padding']}"
                        layer_desc += ")"
                    
                    f.write(layer_desc + "\n")
                else:
                    f.write(f"    {layer}\n")
        elif structure and structure != "None":
            f.write(f"{title}: {structure}\n")
    
    def _write_hyperparameters(self, f, hyperparameters):
        """Write hyper-parameters section."""
        f.write("MODEL HYPER-PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        if hyperparameters:
            for key, value in hyperparameters.items():
                if 'error' in key.lower():
                    f.write(f"ERROR - {key.replace('_', ' ').title()}: {value}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        else:
            f.write("No hyper-parameters extracted\n")
        f.write("\n")
    
    def _write_training_info(self, f, training_info):
        """Write training period information section."""
        f.write("TRAINING PERIOD INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in training_info.items():
            if 'error' in key.lower():
                f.write(f"ERROR - {key.replace('_', ' ').title()}: {value}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
    
    def _write_environment_info(self, f, env_info):
        """Write environment information section."""
        f.write("ENVIRONMENT INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in env_info.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
    
    def _write_episode_summary(self, f, performance_metrics):
        """Write episode summary section."""
        f.write("EPISODE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Steps: {performance_metrics['total_steps']}\n")
        f.write(f"Total Reward: {performance_metrics['total_reward']:.6f}\n")
        f.write(f"Average Reward per Step: {performance_metrics['average_reward_per_step']:.6f}\n")
        f.write(f"Episode Completion: {performance_metrics['episode_completion']}\n")
        f.write("\n")
    
    def _write_step_details(self, f, step_data):
        """Write detailed step-by-step information."""
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
    
    def _write_performance_metrics(self, f, performance_metrics):
        """Write performance metrics section."""
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        
        if 'max_reward' in performance_metrics:
            f.write(f"Max Single Step Reward: {performance_metrics['max_reward']:.6f}\n")
        if 'min_reward' in performance_metrics:
            f.write(f"Min Single Step Reward: {performance_metrics['min_reward']:.6f}\n")
        if 'reward_std' in performance_metrics:
            f.write(f"Reward Standard Deviation: {performance_metrics['reward_std']:.6f}\n")
        if 'portfolio_return' in performance_metrics:
            f.write(f"Portfolio Return: {performance_metrics['portfolio_return']:.2f}%\n")
        
        f.write("\n")
    
    def _write_period_summary(self, f, training_info):
        """Write training vs testing period summary."""
        if 'import_error' not in training_info:
            f.write("PERIOD SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model was trained on data from {training_info['train_start_date']} to {training_info['train_end_date']}\n")
            if isinstance(training_info['training_duration_years'], (int, float)):
                f.write(f"Training period duration: {training_info['training_duration_days']} days ({training_info['training_duration_years']:.2f} years)\n")
            f.write(f"Episode tested on data from {training_info['test_start_date']} to {training_info['test_end_date']}\n")
            f.write("\n")
    
    def _write_footer(self, f):
        """Write report footer."""
        f.write("=" * 80 + "\n")
        f.write(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")


class JSONReportFormatter:
    """Handles JSON report formatting and saving."""
    
    def save_report(self, report_data, output_dir, filename):
        """Save comprehensive JSON report."""
        # Replace .txt with .json for JSON filename
        json_filename = filename.replace('.txt', '.json')
        json_path = os.path.join(output_dir, json_filename)
        
        # Prepare JSON data structure with all detailed information
        json_data = {
            'timestamp': report_data['timestamp'].strftime("%y%m%d_%H%M"),
            'model_info': report_data['model_info'],
            'policy_info': {
                # Basic policy information
                'basic_info': {
                    key: value for key, value in report_data['policy_info'].items()
                    if key in ['policy_class', 'policy_type', 'has_mlp_extractor', 'mlp_extractor_type', 
                              'activation_function', 'network_architecture']
                },
                # Network structures
                'network_structures': {
                    key: value for key, value in report_data['policy_info'].items()
                    if 'structure' in key or 'layers' in key
                },
                # Parameter counts
                'parameter_counts': {
                    key: value for key, value in report_data['policy_info'].items()
                    if 'parameters' in key
                },
                # Network types
                'network_types': {
                    key: value for key, value in report_data['policy_info'].items()
                    if key.endswith('_type') and 'extractor' not in key and 'policy' not in key
                },
                # Action and value network details
                'action_value_networks': {
                    key: value for key, value in report_data['policy_info'].items()
                    if key.startswith('action_') or key.startswith('value_') or key.startswith('critic_') or key.startswith('actor_')
                },
                # Features extractor details
                'features_extractor': {
                    key: value for key, value in report_data['policy_info'].items()
                    if 'features_extractor' in key
                },
                # Errors
                'errors': {
                    key: value for key, value in report_data['policy_info'].items()
                    if 'error' in key.lower()
                }
            },
            'hyperparameters': report_data['hyperparameters'],
            'training_info': report_data['training_info'],
            'environment_info': report_data['environment_info'],
            'episode_summary': {
                'total_steps': report_data['performance_metrics']['total_steps'],
                'total_reward': report_data['performance_metrics']['total_reward'],
                'average_reward_per_step': report_data['performance_metrics']['average_reward_per_step']
            },
            'step_data': report_data['episode_data']['step_data'],
            'performance_metrics': {
                key: value for key, value in report_data['performance_metrics'].items()
                if key not in ['total_steps', 'total_reward', 'average_reward_per_step', 'episode_completion']
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return json_path


class EpisodeReporter:
    """
    Comprehensive episode reporting system for DRL trading models.
    Handles model analysis, data extraction, and report generation.
    """
    
    def __init__(self, output_dir="episode_reports", include_step_details=True):
        """
        Initialize the episode reporter.
        
        Args:
            output_dir: Directory to save reports
            include_step_details: Whether to include detailed step information
        """
        self.output_dir = output_dir
        self.include_step_details = include_step_details
        
        # Initialize extractors
        self.model_extractor = ModelInfoExtractor()
        self.policy_extractor = PolicyInfoExtractor()
        self.hyperparameter_extractor = HyperparameterExtractor()
        self.training_extractor = TrainingInfoExtractor()
        self.environment_extractor = EnvironmentInfoExtractor()
        self.metrics_calculator = PerformanceMetricsCalculator()
        
        # Initialize formatters
        self.text_formatter = TextReportFormatter()
        self.json_formatter = JSONReportFormatter()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, model, environment, step_data, total_reward, 
                       step_count, model_name="Unknown", deterministic=True,
                       formats=['text', 'json'], preserve_existing=False):
        """
        Generate comprehensive episode report in specified formats.
        
        Args:
            model: The trained model used
            environment: Trading environment
            step_data: List of dictionaries containing step information
            total_reward: Final cumulative reward
            step_count: Total number of steps
            model_name: Name of the model
            deterministic: Whether deterministic prediction was used
            formats: List of formats to generate ['text', 'json']
            preserve_existing: If True, adds hyperparameter info to filename to avoid overwriting
        
        Returns:
            Dictionary with paths to generated reports
        """
        
        print("Generating comprehensive episode report...")
        
        # Extract all information
        report_data = self._extract_all_data(
            model, environment, step_data, total_reward, 
            step_count, model_name, deterministic
        )
        
        # Generate filename with optional hyperparameter preservation
        timesteps_formatted = report_data['model_info']['timesteps_formatted']
        
        if preserve_existing:
            # Add key hyperparameters to filename to preserve existing reports
            hyperparam_suffix = self._generate_hyperparam_suffix(report_data['hyperparameters'])
            base_filename = f"Episode_Report_{model_name}_{timesteps_formatted}_{hyperparam_suffix}"
        else:
            # Default filename (may overwrite existing)
            base_filename = f"Episode_Report_{model_name}_{timesteps_formatted}"
        
        # Generate reports in requested formats
        report_paths = {}
        
        if 'text' in formats:
            text_filename = f"{base_filename}.txt"
            report_paths['text'] = self.text_formatter.save_report(
                report_data, self.output_dir, text_filename
            )
            print(f"📊 Text report saved to: {report_paths['text']}")
        
        if 'json' in formats:
            json_filename = f"{base_filename}.txt"  # Will be converted to .json in formatter
            report_paths['json'] = self.json_formatter.save_report(
                report_data, self.output_dir, json_filename
            )
            print(f"📊 JSON report saved to: {report_paths['json']}")
        
        return report_paths
    
    def _generate_hyperparam_suffix(self, hyperparameters):
        """Generate a suffix from key hyperparameters to make filename unique."""
        try:
            # Extract key hyperparameters that commonly change
            key_params = []
            
            if 'learning_rate' in hyperparameters:
                lr = hyperparameters['learning_rate']
                key_params.append(f"lr{lr}".replace('.', '_'))
            
            if 'gamma' in hyperparameters:
                gamma = hyperparameters['gamma']
                key_params.append(f"g{gamma}".replace('.', '_'))
            
            if 'ent_coef' in hyperparameters:
                ent = hyperparameters['ent_coef']
                key_params.append(f"ent{ent}".replace('.', '_'))
            
            if 'n_steps' in hyperparameters:
                steps = hyperparameters['n_steps']
                key_params.append(f"steps{steps}")
            
            if 'vf_coef' in hyperparameters:
                vf = hyperparameters['vf_coef']
                key_params.append(f"vf{vf}".replace('.', '_'))
            
            # Join with underscores, limit length
            suffix = "_".join(key_params)
            # Truncate if too long
            if len(suffix) > 50:
                suffix = suffix[:50]
            
            return suffix if suffix else "custom"
            
        except Exception as e:
            # Fallback to timestamp if hyperparameter extraction fails
            import datetime
            return datetime.datetime.now().strftime("%H%M%S")
    
    def _extract_all_data(self, model, environment, step_data, total_reward, 
                         step_count, model_name, deterministic):
        """Orchestrate data extraction from all sources."""
        
        # Extract information from different sources
        model_info = self.model_extractor.extract(model, model_name, deterministic)
        policy_info = self.policy_extractor.extract(model)
        hyperparameters = self.hyperparameter_extractor.extract(model)
        training_info = self.training_extractor.extract()
        environment_info = self.environment_extractor.extract(environment)
        performance_metrics = self.metrics_calculator.calculate(
            step_data, total_reward, step_count, environment
        )
        
        return {
            'model_info': model_info,
            'policy_info': policy_info,
            'hyperparameters': hyperparameters,
            'training_info': training_info,
            'environment_info': environment_info,
            'performance_metrics': performance_metrics,
            'episode_data': {
                'step_data': step_data,
                'total_reward': total_reward,
                'step_count': step_count
            },
            'timestamp': datetime.datetime.now()
        }
    
    def extract_hyperparameters_only(self, model):
        """
        Utility method to extract only hyperparameters from a model.
        
        Args:
            model: The trained model
            
        Returns:
            Dictionary containing model hyperparameters
        """
        return self.hyperparameter_extractor.extract(model)
    
    def extract_model_info_only(self, model, model_name="Unknown"):
        """
        Utility method to extract only basic model information.
        
        Args:
            model: The trained model
            model_name: Name of the model
            
        Returns:
            Dictionary containing basic model information
        """
        return self.model_extractor.extract(model, model_name)
    
    def set_output_directory(self, output_dir):
        """Change the output directory for reports."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


# Simplified interfaces for easy calling

def save_episode_report(model, environment, step_data, total_reward, step_count, 
                       model_name="Unknown", deterministic=True, preserve_existing=False):
    """
    Backward compatibility wrapper for the original function interface.
    This maintains the same interface as the original function while using the new class system.
    Always includes comprehensive hyperparameters and detailed network architecture.
    
    Args:
        preserve_existing: If True, adds hyperparameter info to filename to avoid overwriting existing reports
    """
    reporter = EpisodeReporter()
    report_paths = reporter.generate_report(
        model, environment, step_data, total_reward, step_count, 
        model_name, deterministic, formats=['text', 'json'], preserve_existing=preserve_existing
    )
    
    # Return paths in the same format as the original function
    return report_paths.get('text'), report_paths.get('json')


def quick_episode_report(model, environment, step_data, total_reward, step_count, model_name="Unknown", preserve_existing=False):
    """
    Super simple one-liner interface for episode reporting.
    Returns just the text report path for simplicity.
    Includes all network architecture details and hyperparameters.
    
    Args:
        preserve_existing: If True, adds hyperparameter info to filename to avoid overwriting existing reports
    """
    return EpisodeReporter().generate_report(
        model, environment, step_data, total_reward, step_count, model_name, preserve_existing=preserve_existing
    )['text']


def full_episode_report(model, environment, step_data, total_reward, step_count, model_name="Unknown", preserve_existing=False):
    """
    Simple interface that returns both report paths with full network details.
    
    Args:
        preserve_existing: If True, adds hyperparameter info to filename to avoid overwriting existing reports
    """
    paths = EpisodeReporter().generate_report(
        model, environment, step_data, total_reward, step_count, model_name, preserve_existing=preserve_existing
    )
    return paths.get('text'), paths.get('json')