"""
Experiment script for varying noise ratios.

This script trains models with different label corruption ratios to study
the relationship between noise and memorization.
"""
import os
import subprocess
import argparse


def run_noise_experiments(base_dir="./experiments/noise_ratio"):
    """
    Run experiments with varying noise ratios.
    
    Tests noise ratios: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    """
    noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for noise in noise_ratios:
        exp_dir = os.path.join(base_dir, f"noise_{noise:.1f}")
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running experiment with noise_ratio = {noise}")
        print(f"{'='*60}\n")
        
        cmd = [
            "python", "train.py",
            "--noise_ratio", str(noise),
            "--checkpoints_dir", exp_dir,
            "--max_epochs", "100",
            "--lr", "0.1",
            "--lr_decay", "0.95",
            "--plot_results"
        ]
        
        subprocess.run(cmd, check=True)
        
    print(f"\n{'='*60}")
    print("All noise ratio experiments completed!")
    print(f"Results saved to: {base_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run noise ratio experiments')
    parser.add_argument('--output_dir', type=str, default='./experiments/noise_ratio',
                       help='Base directory for experiment outputs')
    args = parser.parse_args()
    
    run_noise_experiments(args.output_dir)
