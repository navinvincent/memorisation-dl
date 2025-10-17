"""
Experiment script for comparing explicit regularization techniques.

This script trains models with different regularization strategies:
- No regularization (baseline)
- Weight decay only
- Data augmentation only
- Batch normalization only
- All combined
"""
import os
import subprocess
import argparse


def run_regularization_experiments(base_dir="./experiments/regularization"):
    """
    Run experiments with different regularization techniques.
    """
    experiments = [
        {
            'name': 'no_reg',
            'args': []
        },
        {
            'name': 'weight_decay',
            'args': ['--weight_decay', '1e-4']
        },
        {
            'name': 'augmentation',
            'args': ['--enable_augmentation']
        },
        {
            'name': 'batch_norm',
            'args': ['--enable_batch_norm']
        },
        {
            'name': 'all_reg',
            'args': ['--weight_decay', '1e-4', '--enable_augmentation', '--enable_batch_norm']
        }
    ]
    
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp['name'])
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'='*60}\n")
        
        cmd = [
            "python", "train.py",
            "--noise_ratio", "0.0",  # Clean labels
            "--checkpoints_dir", exp_dir,
            "--max_epochs", "100",
            "--plot_results"
        ] + exp['args']
        
        subprocess.run(cmd, check=True)
        
    print(f"\n{'='*60}")
    print("All regularization experiments completed!")
    print(f"Results saved to: {base_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run regularization experiments')
    parser.add_argument('--output_dir', type=str, default='./experiments/regularization',
                       help='Base directory for experiment outputs')
    args = parser.parse_args()
    
    run_regularization_experiments(args.output_dir)
