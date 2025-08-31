#!/usr/bin/env python3
"""
A2C Episode Report Analysis and Visualization Tool

This script analyzes the most recent A2C episode report and creates comprehensive
visualizations of the neural network outputs, actions, and policies for each trading day.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Stock tickers (29 stocks from DOW_30_TICKER subset)
STOCK_TICKERS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS',
    'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
    'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

def load_episode_report(report_path):
    """Load the JSON episode report."""
    with open(report_path, 'r') as f:
        return json.load(f)

def extract_daily_data(episode_data):
    """Extract daily trading data from episode report."""
    steps = episode_data.get('step_data', [])
    if not steps:
        print("No step_data found in the report")
        return None
    
    daily_data = []
    for step in steps:
        # Handle the action data structure
        action = step['action']
        if isinstance(action, list) and len(action) > 0:
            action = action[0] if isinstance(action[0], list) else action
        
        # Handle policy mean data structure  
        policy_mean = step['policy_mean']
        if isinstance(policy_mean, list) and len(policy_mean) > 0:
            policy_mean = policy_mean[0] if isinstance(policy_mean[0], list) else policy_mean
            
        # Handle policy std data structure
        policy_std = step['policy_std']
        if isinstance(policy_std, list) and len(policy_std) > 0:
            policy_std = policy_std[0] if isinstance(policy_std[0], list) else policy_std
        
        step_data = {
            'step': step['step'],
            'day': step['day'],
            'action': np.array(action),
            'policy_mean': np.array(policy_mean),
            'policy_std': np.array(policy_std),
            'reward': float(step['reward']),
            'done': step['done'] == 'True'
        }
        daily_data.append(step_data)
    
    return daily_data

def create_action_heatmap(daily_data, save_path=None):
    """Create a heatmap of daily actions for all stocks."""
    # Extract actions for all days
    actions_matrix = np.array([day['action'] for day in daily_data])
    days = [day['day'] for day in daily_data]
    
    plt.figure(figsize=(16, 10))
    
    # Create heatmap
    im = plt.imshow(actions_matrix.T, aspect='auto', cmap='RdBu_r', 
                    vmin=-1, vmax=1, interpolation='nearest')
    
    # Customization
    plt.colorbar(im, label='Action Value', shrink=0.8)
    plt.title('A2C Daily Trading Actions Heatmap\n(Blue=Sell, Red=Buy)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Trading Day', fontsize=14)
    plt.ylabel('Stock Ticker', fontsize=14)
    
    # Set tickers on y-axis
    plt.yticks(range(len(STOCK_TICKERS)), STOCK_TICKERS, fontsize=10)
    
    # Set day labels on x-axis (every 10th day)
    x_ticks = range(0, len(days), max(1, len(days)//20))
    plt.xticks(x_ticks, [days[i] for i in x_ticks], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action heatmap saved to: {save_path}")
    
    plt.show()

def create_policy_uncertainty_plot(daily_data, save_path=None):
    """Create a plot showing policy uncertainty (standard deviation) over time."""
    days = [day['day'] for day in daily_data]
    policy_stds = np.array([day['policy_std'] for day in daily_data])
    
    # Calculate average uncertainty per day and per stock
    avg_uncertainty_per_day = np.mean(policy_stds, axis=1)
    avg_uncertainty_per_stock = np.mean(policy_stds, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Average uncertainty over time
    ax1.plot(days, avg_uncertainty_per_day, linewidth=2, color='darkblue', alpha=0.8)
    ax1.fill_between(days, avg_uncertainty_per_day, alpha=0.3, color='lightblue')
    ax1.set_title('Policy Uncertainty Over Time (Average Across All Stocks)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Average Policy Standard Deviation')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty by stock
    ax2.bar(range(len(STOCK_TICKERS)), avg_uncertainty_per_stock, 
            color='steelblue', alpha=0.7)
    ax2.set_title('Average Policy Uncertainty by Stock', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Stock Ticker')
    ax2.set_ylabel('Average Policy Standard Deviation')
    ax2.set_xticks(range(len(STOCK_TICKERS)))
    ax2.set_xticklabels(STOCK_TICKERS, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy uncertainty plot saved to: {save_path}")
    
    plt.show()

def create_reward_analysis(daily_data, save_path=None):
    """Create reward analysis visualizations."""
    days = [day['day'] for day in daily_data]
    rewards = [day['reward'] for day in daily_data]
    cumulative_rewards = np.cumsum(rewards)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Daily rewards
    colors = ['red' if r < 0 else 'green' for r in rewards]
    ax1.bar(days, rewards, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('Daily Rewards from A2C Trading Actions', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Daily Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative rewards
    ax2.plot(days, cumulative_rewards, linewidth=2, color='darkgreen')
    ax2.fill_between(days, cumulative_rewards, alpha=0.3, color='lightgreen')
    ax2.set_title('Cumulative Rewards Over Trading Period', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Cumulative Reward')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reward analysis plot saved to: {save_path}")
    
    plt.show()

def create_action_distribution_analysis(daily_data, save_path=None):
    """Analyze the distribution of actions across different ranges."""
    all_actions = np.concatenate([day['action'] for day in daily_data])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Action distribution histogram
    ax1.hist(all_actions, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Neutral (0)')
    ax1.axvline(x=1, color='green', linestyle='--', alpha=0.8, label='Max Buy (1)')
    ax1.axvline(x=-1, color='red', linestyle='--', alpha=0.8, label='Max Sell (-1)')
    ax1.set_title('Distribution of All Trading Actions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Action Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Action categories pie chart
    buy_actions = np.sum(all_actions > 0.1)
    sell_actions = np.sum(all_actions < -0.1)
    neutral_actions = np.sum(np.abs(all_actions) <= 0.1)
    
    labels = ['Buy (>0.1)', 'Sell (<-0.1)', 'Neutral (±0.1)']
    sizes = [buy_actions, sell_actions, neutral_actions]
    colors = ['lightgreen', 'lightcoral', 'lightgray']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Action Categories Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action distribution analysis saved to: {save_path}")
    
    plt.show()

def create_stock_specific_analysis(daily_data, stock_indices=None, save_path=None):
    """Create detailed analysis for specific stocks."""
    if stock_indices is None:
        # Default to first 6 stocks for visualization
        stock_indices = [0, 1, 2, 3, 4, 5]
    
    days = [day['day'] for day in daily_data]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, stock_idx in enumerate(stock_indices):
        if i >= len(axes):
            break
            
        # Extract data for this stock
        actions = [day['action'][stock_idx] for day in daily_data]
        policy_means = [day['policy_mean'][stock_idx] for day in daily_data]
        policy_stds = [day['policy_std'][stock_idx] for day in daily_data]
        
        ax = axes[i]
        
        # Plot action and policy mean
        ax.plot(days, actions, label='Actual Action', linewidth=2, color='blue')
        ax.plot(days, policy_means, label='Policy Mean', linewidth=2, color='red', alpha=0.7)
        
        # Add uncertainty bands
        upper_bound = np.array(policy_means) + np.array(policy_stds)
        lower_bound = np.array(policy_means) - np.array(policy_stds)
        ax.fill_between(days, lower_bound, upper_bound, alpha=0.2, color='red', 
                       label='Policy Std (±1σ)')
        
        ax.set_title(f'{STOCK_TICKERS[stock_idx]} - Action vs Policy', fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Action Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-6, 6)  # Set consistent scale
    
    plt.suptitle('Individual Stock Analysis: Actions vs Policy Outputs', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stock-specific analysis saved to: {save_path}")
    
    plt.show()

def generate_summary_statistics(daily_data):
    """Generate and print summary statistics."""
    print("\n" + "="*80)
    print("A2C EPISODE ANALYSIS SUMMARY")
    print("="*80)
    
    # Basic info
    print(f"Total Trading Days: {len(daily_data)}")
    print(f"Stock Universe: {len(STOCK_TICKERS)} stocks")
    
    # Reward statistics
    rewards = [day['reward'] for day in daily_data]
    print(f"\nREWARD STATISTICS:")
    print(f"  Total Cumulative Reward: {sum(rewards):.4f}")
    print(f"  Average Daily Reward: {np.mean(rewards):.4f}")
    print(f"  Reward Std Dev: {np.std(rewards):.4f}")
    print(f"  Best Day Reward: {max(rewards):.4f}")
    print(f"  Worst Day Reward: {min(rewards):.4f}")
    
    # Action statistics
    all_actions = np.concatenate([day['action'] for day in daily_data])
    print(f"\nACTION STATISTICS:")
    print(f"  Total Actions: {len(all_actions)}")
    print(f"  Action Range: [{np.min(all_actions):.4f}, {np.max(all_actions):.4f}]")
    print(f"  Mean Action: {np.mean(all_actions):.4f}")
    print(f"  Action Std Dev: {np.std(all_actions):.4f}")
    
    # Policy statistics
    all_policy_stds = np.concatenate([day['policy_std'] for day in daily_data])
    print(f"\nPOLICY UNCERTAINTY STATISTICS:")
    print(f"  Average Policy Std: {np.mean(all_policy_stds):.4f}")
    print(f"  Policy Std Range: [{np.min(all_policy_stds):.4f}, {np.max(all_policy_stds):.4f}]")
    
    # Trading behavior
    buy_actions = np.sum(all_actions > 0.1)
    sell_actions = np.sum(all_actions < -0.1)
    neutral_actions = np.sum(np.abs(all_actions) <= 0.1)
    
    print(f"\nTRADING BEHAVIOR:")
    print(f"  Buy Actions (>0.1): {buy_actions} ({buy_actions/len(all_actions)*100:.1f}%)")
    print(f"  Sell Actions (<-0.1): {sell_actions} ({sell_actions/len(all_actions)*100:.1f}%)")
    print(f"  Neutral Actions (±0.1): {neutral_actions} ({neutral_actions/len(all_actions)*100:.1f}%)")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Analyze A2C Episode Report')
    parser.add_argument('--report', type=str, 
                       default='episode_reports/Episode_Report_A2C_1e7_lr5e-05_g0_99_ent0_1_steps16_vf0_25.json',
                       help='Path to episode report JSON file')
    parser.add_argument('--output-dir', type=str, default='episode_analysis_plots',
                       help='Directory to save plots')
    parser.add_argument('--show-plots', action='store_true', default=True,
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load episode report
    print(f"Loading episode report: {args.report}")
    episode_data = load_episode_report(args.report)
    
    # Extract daily data
    print("Extracting daily trading data...")
    daily_data = extract_daily_data(episode_data)
    
    if not daily_data:
        print("Error: Could not extract daily data from episode report")
        return
    
    # Generate summary statistics
    generate_summary_statistics(daily_data)
    
    # Create visualizations
    print(f"\nGenerating visualizations (saving to {output_dir})...")
    
    create_action_heatmap(daily_data, output_dir / 'action_heatmap.png')
    create_policy_uncertainty_plot(daily_data, output_dir / 'policy_uncertainty.png')
    create_reward_analysis(daily_data, output_dir / 'reward_analysis.png')
    create_action_distribution_analysis(daily_data, output_dir / 'action_distribution.png')
    create_stock_specific_analysis(daily_data, save_path=output_dir / 'stock_specific_analysis.png')
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()