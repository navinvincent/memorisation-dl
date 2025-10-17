"""
Visualization utilities for plotting training statistics
"""
import matplotlib.pyplot as plt
import numpy as np
import jax


def plot_training_statistics(metrics, last_epoch, lr_scheduler, steps_per_epoch, save_path=None):
    """
    Plot training and test loss, accuracy, and learning rate schedule.
    
    Args:
        metrics: dict containing 'train' and 'test' metrics
        last_epoch: final epoch number
        lr_scheduler: learning rate scheduler function
        steps_per_epoch: number of batches per epoch
        save_path: optional path to save the figure
    """
    fig, axs = plt.subplots(3, figsize=(10, 10), sharex=True)
    epochs = np.arange(len(metrics['test']['loss'])) * 10
    if last_epoch % 10 != 0:
        epochs[-1] = last_epoch

    axs[0].plot(epochs, metrics['train']['loss'], label='train loss')
    axs[0].plot(epochs, metrics['test']['loss'], label='test loss')
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('loss')
    
    axs[1].plot(epochs, metrics['train']['accuracy'], label='train acc')
    axs[1].plot(epochs, metrics['test']['accuracy'], label='test acc')
    axs[1].legend(loc='center right')
    axs[1].set_ylabel('accuracy')
    axs[1].set_ylim(0., 1.0)

    # plot learning rate schedule
    lr_schedule = [jax.device_get(lr_scheduler(epoch * steps_per_epoch)).item() for epoch in epochs]

    axs[2].plot(epochs, lr_schedule)
    axs[2].set_ylabel('learning rate')
    axs[2].set_xlabel('epochs')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig
