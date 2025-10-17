"""
Analysis utilities for comparing experiment results.

This script helps analyze and compare results from different experiments.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch


def load_checkpoint_metrics(checkpoint_dir):
    """
    Load the latest checkpoint from a directory and extract basic info.
    
    Args:
        checkpoint_dir: directory containing checkpoint files
        
    Returns:
        dict with checkpoint info
    """
    checkpoints = glob(os.path.join(checkpoint_dir, "*.pickle"))
    if not checkpoints:
        return None
    
    # Find the latest checkpoint (excluding 'init')
    numeric_checkpoints = []
    for cp in checkpoints:
        if 'init' not in cp:
            try:
                epoch = int(cp.replace('.pickle', '').split('_')[-1])
                numeric_checkpoints.append((epoch, cp))
            except:
                continue
    
    if not numeric_checkpoints:
        return None
    
    latest_epoch, latest_checkpoint = max(numeric_checkpoints, key=lambda x: x[0])
    state_dict = torch.load(latest_checkpoint)
    
    return {
        'epoch': state_dict['epoch'],
        'checkpoint_path': latest_checkpoint
    }


def compare_noise_experiments(base_dir="./experiments/noise_ratio", save_path=None):
    """
    Compare results across different noise ratio experiments.
    
    Args:
        base_dir: base directory containing noise ratio experiments
        save_path: path to save comparison plot
    """
    noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}
    
    for noise in noise_ratios:
        exp_dir = os.path.join(base_dir, f"noise_{noise:.1f}")
        if os.path.exists(exp_dir):
            checkpoint_info = load_checkpoint_metrics(exp_dir)
            if checkpoint_info:
                results[noise] = checkpoint_info
    
    if not results:
        print(f"No experiment results found in {base_dir}")
        return
    
    print("\nNoise Ratio Experiment Results:")
    print("=" * 60)
    for noise, info in sorted(results.items()):
        print(f"Noise Ratio: {noise:.1f}")
        print(f"  Final Epoch: {info['epoch']}")
        print(f"  Checkpoint: {info['checkpoint_path']}")
        print()
    
    return results


def print_experiment_summary(checkpoint_dir):
    """
    Print a summary of an experiment from its checkpoint directory.
    
    Args:
        checkpoint_dir: directory containing experiment checkpoints
    """
    info = load_checkpoint_metrics(checkpoint_dir)
    
    if info is None:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\nExperiment: {os.path.basename(checkpoint_dir)}")
    print("=" * 60)
    print(f"Final Epoch: {info['epoch']}")
    print(f"Checkpoint: {info['checkpoint_path']}")
    
    # Check for plot
    plot_path = os.path.join(checkpoint_dir, 'training_statistics.png')
    if os.path.exists(plot_path):
        print(f"Training plot: {plot_path}")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--type', type=str, choices=['noise', 'regularization', 'single'],
                       default='single', help='Type of experiment to analyze')
    
    args = parser.parse_args()
    
    if args.type == 'noise':
        compare_noise_experiments(args.experiment_dir)
    elif args.type == 'regularization':
        # Can be extended for regularization comparison
        print_experiment_summary(args.experiment_dir)
    else:
        print_experiment_summary(args.experiment_dir)
