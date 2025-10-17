"""


This script runs a simple memorization experiment and shows how to use
the  codebase.
"""
import os
import jax

from configs.default_config import *
from src.data.dataset import get_dataloaders
from src.training.trainer import (
    create_train_state, train_epoch, eval_model, 
    get_lr_scheduler, stopping_criterion
)
from src.utils.visualization import plot_training_statistics


def run_simple_experiment(noise_ratio=1.0, max_epochs=50, output_dir="./example_output"):
    """
    Run a simple memorization experiment.
    
    Args:
        noise_ratio: ratio of labels to corrupt (0.0 to 1.0)
        max_epochs: maximum number of training epochs
        output_dir: directory to save results
    """
    print("=" * 70)
    print(f"Running Simple Memorization Experiment")
    print(f"Noise Ratio: {noise_ratio}")
    print(f"Max Epochs: {max_epochs}")
    print("=" * 70)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        enable_augmentation=ENABLE_AUGMENTATION,
        cifar10_mean=CIFAR10_MEAN,
        cifar10_std=CIFAR10_STD,
        noise_ratio=noise_ratio,
        seed=SEED
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    # Initialize model
    print("Initializing ResNet-18 model...")
    rng = jax.random.PRNGKey(SEED)
    state = create_train_state(
        rng, NUM_CLASSES, LR, LR_STEP, 
        LR_DECAY, len(train_loader), NUM_FILTERS
    )
    print(f"Model parameters initialized")
    print()
    
    # Training loop
    metrics = {
        'train': {'loss': [], 'accuracy': []},
        'test': {'loss': [], 'accuracy': []}
    }
    
    print(f"Starting training for up to {max_epochs} epochs...")
    print(f"Stopping criterion: train accuracy >= 0.999")
    print()
    
    for epoch in range(max_epochs):
        # Train for one epoch
        state, train_metrics = train_epoch(
            state, train_loader, epoch, NUM_CLASSES, 
            WEIGHT_DECAY, NUM_FILTERS
        )
        
        # Check stopping criterion
        if stopping_criterion(train_metrics['accuracy']):
            print(f"\nStopping criterion reached at epoch {epoch}!")
            test_metrics = eval_model(epoch, state.params, test_loader, NUM_CLASSES, NUM_FILTERS)
            for key in train_metrics:
                metrics['train'][key].append(train_metrics[key])
                metrics['test'][key].append(test_metrics[key])
            last_epoch = epoch
            break
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            test_metrics = eval_model(epoch, state.params, test_loader, NUM_CLASSES, NUM_FILTERS)
            for key in train_metrics:
                metrics['train'][key].append(train_metrics[key])
                metrics['test'][key].append(test_metrics[key])
    else:
        last_epoch = max_epochs - 1
        test_metrics = eval_model(last_epoch, state.params, test_loader, NUM_CLASSES, NUM_FILTERS)
        for key in test_metrics:
            metrics['train'][key].append(train_metrics[key])
            metrics['test'][key].append(test_metrics[key])
    
    # Print results
    print()
    print("=" * 70)
    print("Experiment Results")
    print("=" * 70)
    print(f"Final Epoch: {last_epoch}")
    print(f"Final Train Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Train Loss: {train_metrics['loss']:.4f}")
    print(f"Final Test Loss: {test_metrics['loss']:.4f}")
    print()
    
    # Generate plot
    print("Generating training plot...")
    lr_scheduler = get_lr_scheduler(LR, LR_STEP, LR_DECAY, len(train_loader))
    plot_path = os.path.join(output_dir, 'training_plot.png')
    plot_training_statistics(
        metrics, last_epoch, lr_scheduler, 
        len(train_loader), save_path=plot_path
    )
    
    print()
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    return state, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a simple memorization experiment')
    parser.add_argument('--noise_ratio', type=float, default=1.0,
                       help='Ratio of labels to corrupt (0.0 to 1.0)')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./example_output',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    run_simple_experiment(
        noise_ratio=args.noise_ratio,
        max_epochs=args.max_epochs,
        output_dir=args.output_dir
    )
