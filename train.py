"""
Main training script for memorization experiments.

This script trains a ResNet-18 model on CIFAR-10 with configurable label corruption
to study the memorization vs. generalization behavior of deep neural networks.

Usage:
    python train.py --noise_ratio 1.0 --max_epochs 100 --lr 0.1
    python train.py --config configs/custom_config.py
"""
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np

from configs.default_config import *
from src.data.dataset import get_dataloaders
from src.training.trainer import (
    create_train_state, train_epoch, eval_model, 
    save_checkpoint, load_checkpoint, load_train_state,
    stopping_criterion, get_lr_scheduler
)
from src.utils.visualization import plot_training_statistics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10 with label corruption')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                       help='Path to CIFAR-10 dataset')
    parser.add_argument('--checkpoints_dir', type=str, default=CHECKPOINTS_DIR,
                       help='Directory to save checkpoints')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS,
                       help='Maximum number of training epochs')
    parser.add_argument('--seed', type=int, default=SEED,
                       help='Random seed')
    
    # Model arguments
    parser.add_argument('--num_filters', type=int, default=NUM_FILTERS,
                       help='Number of filters in first ResNet layer')
    parser.add_argument('--enable_batch_norm', action='store_true',
                       default=ENABLE_BATCH_NORM,
                       help='Enable batch normalization')
    
    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=LR,
                       help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=LR_DECAY,
                       help='Learning rate decay factor')
    parser.add_argument('--lr_step', type=int, default=LR_STEP,
                       help='Learning rate decay step')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                       help='Weight decay coefficient')
    
    # Data augmentation and noise
    parser.add_argument('--noise_ratio', type=float, default=NOISE_RATIO,
                       help='Ratio of labels to corrupt (0.0 to 1.0)')
    parser.add_argument('--enable_augmentation', action='store_true',
                       default=ENABLE_AUGMENTATION,
                       help='Enable data augmentation')
    
    # Other arguments
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--plot_results', action='store_true', default=True,
                       help='Plot training results')
    
    return parser.parse_args()


def train_and_evaluate(args):
    """
    Main training and evaluation loop.
    
    Args:
        args: parsed command line arguments
        
    Returns:
        state: final training state
        metrics: training and test metrics
        last_epoch: final epoch number
    """
    # Set JAX platform
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # Get dataloaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        enable_augmentation=args.enable_augmentation,
        cifar10_mean=CIFAR10_MEAN,
        cifar10_std=CIFAR10_STD,
        noise_ratio=args.noise_ratio,
        seed=args.seed
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Label corruption ratio: {args.noise_ratio}")
    
    # Initialize model
    rng = jax.random.PRNGKey(args.seed)
    
    # Load from checkpoint if resuming
    if args.resume:
        state_dict = load_checkpoint(args.checkpoints_dir)
        if state_dict is None:
            print("No checkpoint found, starting from scratch")
            state = create_train_state(
                rng, NUM_CLASSES, args.lr, args.lr_step, 
                args.lr_decay, len(train_loader), args.num_filters
            )
            save_checkpoint(args.checkpoints_dir, state, 'init')
            start_epoch = 0
        else:
            state = load_train_state(
                state_dict, args.lr, args.lr_step, args.lr_decay, 
                len(train_loader), NUM_CLASSES, args.num_filters
            )
            start_epoch = state_dict['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    else:
        state = create_train_state(
            rng, NUM_CLASSES, args.lr, args.lr_step, 
            args.lr_decay, len(train_loader), args.num_filters
        )
        save_checkpoint(args.checkpoints_dir, state, 'init')
        start_epoch = 0
    
    # Initialize metrics storage
    metrics = {
        'train': {
            'loss': [],
            'accuracy': [],
        },
        'test': {
            'loss': [],
            'accuracy': [],
        },
    }
    
    # Training loop
    print(f"\nStarting training for {args.max_epochs} epochs...")
    print(f"Stopping criterion: train accuracy >= 0.999")
    
    for epoch in range(start_epoch, args.max_epochs):
        state, train_metrics = train_epoch(
            state, train_loader, epoch, NUM_CLASSES, 
            args.weight_decay, args.num_filters
        )
        
        # Check stopping criterion
        if stopping_criterion(train_metrics['accuracy']):
            print(f"\nStopping criterion reached: train accuracy = {train_metrics['accuracy']:.4f}")
            test_metrics = eval_model(epoch, state.params, test_loader, NUM_CLASSES, args.num_filters)
            save_checkpoint(args.checkpoints_dir, state, epoch)
            for key in train_metrics:
                metrics['train'][key].append(train_metrics[key])
                metrics['test'][key].append(test_metrics[key])
            last_epoch = epoch
            break
        
        # Evaluate and save checkpoint every 10 epochs
        if epoch % 10 == 0:
            test_metrics = eval_model(epoch, state.params, test_loader, NUM_CLASSES, args.num_filters)
            save_checkpoint(args.checkpoints_dir, state, epoch)
            for key in train_metrics:
                metrics['train'][key].append(train_metrics[key])
                metrics['test'][key].append(test_metrics[key])
    else:
        # If we didn't break (reach stopping criterion)
        last_epoch = args.max_epochs - 1
        save_checkpoint(args.checkpoints_dir, state, last_epoch)
        test_metrics = eval_model(last_epoch, state.params, test_loader, NUM_CLASSES, args.num_filters)
        for key in test_metrics:
            metrics['train'][key].append(train_metrics[key])
            metrics['test'][key].append(test_metrics[key])
    
    print("\nTraining completed!")
    print(f"Final train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    
    return state, metrics, last_epoch


def main():
    """Main entry point"""
    args = parse_args()
    
    # Train model
    state, metrics, last_epoch = train_and_evaluate(args)
    
    # Plot results if requested
    if args.plot_results:
        print("\nGenerating training plots...")
        lr_scheduler = get_lr_scheduler(args.lr, args.lr_step, args.lr_decay, 
                                        steps_per_epoch=50000 // args.batch_size)
        
        plot_path = os.path.join(args.checkpoints_dir, 'training_statistics.png')
        plot_training_statistics(
            metrics, last_epoch, lr_scheduler, 
            steps_per_epoch=50000 // args.batch_size,
            save_path=plot_path
        )


if __name__ == '__main__':
    main()
