import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Supported algorithms and their result paths
ALGO_PATHS = {
    'a2c': '/deac/csc/classes/csc790/jianb21/Ensemble_stockTrading_2020/results/a2c/progress.csv',
    # 'ppo': 'scripts/results/ppo/progress.csv',
    # 'ddpg': 'scripts/results/ddpg/progress.csv',
}

# Define the metrics to plot for each algorithm
ALGO_METRICS = {
    'a2c': [
        ('train/reward', 'Reward'),
        ('train/policy_loss', 'Policy Loss'),
        ('train/value_loss', 'Value Loss'),
        ('train/explained_variance', 'Explained Variance'),
        ('train/entropy_loss', 'Entropy Loss'),
        ('train/std', 'Action Std'),
        ('time/fps', 'FPS'),
    ],
    # 'ppo': [
    #     ('train/reward', 'Reward'),
    #     ('train/loss', 'Policy Loss'),
    #     ('train/value_loss', 'Value Loss'),
    #     ('train/explained_variance', 'Explained Variance'),
    #     ('train/entropy_loss', 'Entropy Loss'),
    #     ('train/std', 'Action Std'),
    #     ('time/fps', 'FPS'),
    #     ('train/learning_rate', 'Learning Rate'),
    #     ('train/n_updates', 'N Updates'),
    # ],
    # 'ddpg': [
    #     ('train/reward', 'Reward'),
    #     ('train/actor_loss', 'Actor Loss'),
    #     ('train/critic_loss', 'Critic Loss'),
    #     ('time/fps', 'FPS'),
    #     ('train/learning_rate', 'Learning Rate'),
    #     ('train/n_updates', 'N Updates'),
    # ],
}

# X-axis for each algorithm
ALGO_X = {
    'a2c': 'time/total_timesteps',
    # 'ppo': 'time/total_timesteps',
    # 'ddpg': 'time/total_timesteps',
}

def load_results(algo):
    path = ALGO_PATHS[algo]
    if not os.path.exists(path):
        print(f"[Warning] File not found for {algo}: {path}")
        return None
    df = pd.read_csv(path)
    return df

def plot_individual_metrics(algo, df, save_dir='results/a2c/graphs'):
    metrics = ALGO_METRICS[algo]
    x_col = ALGO_X[algo]
    n = len(metrics)
    ncols = 9
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18*ncols, 12*nrows))
    axes = axes.flatten()
    for i, (col, title) in enumerate(metrics):
        if col in df.columns:
            axes[i].plot(df[x_col], df[col], label=title)
            axes[i].set_title(title)
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(title)
            axes[i].grid(True, alpha=0.5)
        else:
            axes[i].set_visible(False)
    plt.suptitle(f'{algo.upper()} Training Metrics', fontsize=30, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(save_dir, f'{algo}_training_metrics.png')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"Saved {algo.upper()} metrics plot to {out_path}")

def plot_comparative_metric(metric, algos, dfs, save_dir='scripts/graphs'):
    plt.figure(figsize=(10,6))
    for algo, df in dfs.items():
        if df is not None and metric in df.columns:
            x_col = ALGO_X[algo]
            plt.plot(df[x_col], df[metric], label=algo.upper())
    plt.title(f'Comparative {metric}')
    plt.xlabel('Timesteps')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(save_dir, f'compare_{metric}.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved comparative plot for {metric} to {out_path}")

def print_summary(algo, df):
    print(f"\n=== {algo.upper()} Training Summary ===")
    if 'train/reward' in df.columns:
        print(f"Final reward: {df['train/reward'].iloc[-1]:.4f}")
        print(f"Average reward: {df['train/reward'].mean():.4f}")
        print(f"Best reward: {df['train/reward'].max():.4f}")
    if 'time/total_timesteps' in df.columns:
        print(f"Total timesteps: {df['time/total_timesteps'].max():,}")
    if 'time/time_elapsed' in df.columns:
        print(f"Training time: {df['time/time_elapsed'].max():.1f} seconds ({df['time/time_elapsed'].max()/3600:.1f} hours)")
    if 'time/fps' in df.columns:
        print(f"Training speed: {df['time/fps'].mean():.1f} FPS average")
    if 'train/explained_variance' in df.columns:
        print(f"Final explained variance: {df['train/explained_variance'].iloc[-1]:.3f}")

def main():
    algos = ['a2c'] #, 'ppo', 'ddpg'
    dfs = {algo: load_results(algo) for algo in algos}
    # Plot individual metrics
    for algo, df in dfs.items():
        if df is not None:
            plot_individual_metrics(algo, df)
            print_summary(algo, df)
    # Plot comparative reward
    # plot_comparative_metric('train/reward', algos, dfs)s
    # Optionally, plot other comparative metrics
    # for metric in ['train/value_loss', 'train/policy_loss', 'train/actor_loss', 'train/critic_loss']:
        # plot_comparative_metric(metric, algos, dfs)

if __name__ == '__main__':
    main()