"""
Training utilities and metrics computation
"""
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
import os
from glob import glob
import torch

from ..models.resnet import ResNet18


def compute_metrics(logits, labels):
    """
    Compute loss and accuracy from model predictions.
    
    Args:
        logits: unnormalized model predictions
        labels: ground-truth labels
        
    Returns:
        dict with 'loss' and 'accuracy' keys
    """
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels).mean()
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def get_lr_scheduler(base_lr, lr_step, lr_decay, steps_per_epoch):
    """
    Initialize exponential learning rate decay scheduler.
    
    Args:
        base_lr: initial learning rate
        lr_step: number of training steps after which the learning rate is updated
        lr_decay: learning rate update coefficient
        steps_per_epoch: number of SGD steps in one training epoch
        
    Returns:
        optax learning rate scheduler
    """
    scheduler = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=steps_per_epoch / lr_step,
        decay_rate=lr_decay,
        staircase=False
    )
    return scheduler


def get_piecewise_lr_scheduler(base_lr, steps_per_epoch, boundaries_and_scales):
    """
    Initialize piecewise constant learning rate scheduler.
    
    Args:
        base_lr: initial learning rate
        steps_per_epoch: number of SGD steps in one training epoch
        boundaries_and_scales: dict mapping epoch boundaries to scale factors
        
    Returns:
        optax learning rate scheduler
    """
    scheduler = optax.piecewise_constant_schedule(
        init_value=base_lr, 
        boundaries_and_scales=boundaries_and_scales
    )
    return scheduler


def stopping_criterion(accuracy, threshold=0.999):
    """
    Check whether target training accuracy is reached.
    
    Args:
        accuracy: current training accuracy
        threshold: accuracy threshold for stopping
        
    Returns:
        True if accuracy >= threshold, False otherwise
    """
    return accuracy >= threshold


def create_train_state(rng, num_classes, lr, lr_step, lr_decay, steps_per_epoch, num_filters=16):
    """
    Initialize network and optimizer.
    
    Args:
        rng: JAX random key
        num_classes: number of output classes
        lr: learning rate
        lr_step: learning rate decay step
        lr_decay: learning rate decay factor
        steps_per_epoch: number of batches per epoch
        num_filters: number of filters in first ResNet layer
        
    Returns:
        TrainState object
    """
    net = ResNet18(num_classes=num_classes, num_filters=num_filters)
    params = net.init(rng, jnp.ones((1, 32, 32, 3)))['params']
    
    learning_rate_scheduler = get_lr_scheduler(lr, lr_step, lr_decay, steps_per_epoch)
    tx = optax.sgd(learning_rate_scheduler, momentum=0.9, nesterov=False)
    return train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, num_classes, weight_decay, num_filters=16):
    """
    Perform one training step.
    
    Args:
        state: training state
        batch: batch of data with 'image' and 'label' keys
        num_classes: number of output classes
        weight_decay: L2 regularization coefficient
        num_filters: number of filters in first ResNet layer
        
    Returns:
        updated state and metrics
    """
    def compute_loss(params):
        """Cross-entropy loss with weight decay"""
        logits = ResNet18(num_classes=num_classes, num_filters=num_filters).apply(
            {'params': params}, batch['image'])
        loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits=logits, 
                labels=jax.nn.one_hot(batch['label'], num_classes)))
        weight_penalty_params = jax.tree.leaves(params)
        weight_l2 = sum(
            [jnp.sum(x ** 2) for x in weight_penalty_params if x.ndim > 1])
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss, logits
    
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics


@jax.jit
def eval_step(params, batch, num_classes, num_filters=16):
    """
    Evaluate model on a batch.
    
    Args:
        params: model parameters
        batch: batch of data with 'image' and 'label' keys
        num_classes: number of output classes
        num_filters: number of filters in first ResNet layer
        
    Returns:
        metrics dict
    """
    logits = ResNet18(num_classes=num_classes, num_filters=num_filters).apply(
        {'params': params}, batch['image'], train=False)
    return compute_metrics(logits=logits, labels=batch['label'])


def train_epoch(state, train_loader, epoch, num_classes, weight_decay, num_filters=16):
    """
    Train for one epoch.
    
    Args:
        state: training state
        train_loader: training data loader
        epoch: current epoch number
        num_classes: number of output classes
        weight_decay: L2 regularization coefficient
        num_filters: number of filters in first ResNet layer
        
    Returns:
        updated state and epoch metrics
    """
    batch_metrics = []
    for input, target in train_loader:
        batch = {
            'image': input,
            'label': target,
        }

        state, train_metrics_ep = train_step(state, batch, num_classes, weight_decay, num_filters)
        batch_metrics.append(train_metrics_ep)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print(f"epoch: {epoch}, train loss: {epoch_metrics_np['loss']:.4f}, "
          f"train accuracy: {epoch_metrics_np['accuracy']:.4f}")
    return state, epoch_metrics_np


def eval_model(epoch, params, test_loader, num_classes, num_filters=16):
    """
    Evaluate model on test set.
    
    Args:
        epoch: current epoch number
        params: model parameters
        test_loader: test data loader
        num_classes: number of output classes
        num_filters: number of filters in first ResNet layer
        
    Returns:
        epoch metrics dict
    """
    batch_metrics = []
    for input, target in test_loader:
        batch = {
            'image': input,
            'label': target,
        }

        metrics = eval_step(params, batch, num_classes, num_filters)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print(f"epoch: {epoch}, test loss: {epoch_metrics_np['loss']:.4f}, "
          f"test accuracy: {epoch_metrics_np['accuracy']:.4f}")
    return epoch_metrics_np


def save_checkpoint(savedir, train_state, epoch):
    """Save training state to checkpoint file"""
    state_dict = {
        'epoch': epoch,
        'state': train_state.params,
    }
    if isinstance(epoch, int):
        save_path = os.path.join(savedir, f"resnet18_{epoch}.pickle")
    else:
        save_path = os.path.join(savedir, f"resnet18_{epoch}.pickle")
    torch.save(state_dict, save_path)
    print(f"Saving model checkpoint to {save_path}.")


def try_cast(maybe_number):
    """Try to cast string to integer"""
    try:
        number = int(maybe_number)
        return number
    except:
        return None


def load_checkpoint(savedir):
    """
    Load the latest checkpoint from savedir.
    
    Returns:
        state_dict or None if no checkpoint found
    """
    save_path = glob(os.path.join(savedir, '*.pickle'))
    path_dict = {}
    path_sections = map(
        lambda x: x.replace(".pickle", "").split("_")[-1],
        save_path)
    for i, maybe_num in enumerate(path_sections):
        num = try_cast(maybe_num)
        if num is not None:
            path_dict[num] = save_path[i]
    if len(path_dict) != 0:
        latest_checkpoint = sorted(path_dict.items(),
                    key=lambda x: x[1])[-1][1]
        state_dict = torch.load(latest_checkpoint)
        print(f"Loading model from checkpoint {latest_checkpoint}.")
        return state_dict
    else:
        return None


def load_train_state(state_dict, lr, lr_step, lr_decay, steps_per_epoch, num_classes, num_filters=16):
    """
    Load training state from checkpoint.
    
    Args:
        state_dict: saved state dictionary
        lr: learning rate
        lr_step: learning rate decay step
        lr_decay: learning rate decay factor
        steps_per_epoch: number of batches per epoch
        num_classes: number of output classes
        num_filters: number of filters in first ResNet layer
        
    Returns:
        TrainState object
    """
    net = ResNet18(num_classes=num_classes, num_filters=num_filters)
    params = state_dict['state']
    lr_steps = state_dict['epoch'] * steps_per_epoch
    learning_rate_scheduler = get_lr_scheduler(lr, lr_step, lr_decay, steps_per_epoch)
    tx = optax.sgd(learning_rate_scheduler, momentum=0.9, nesterov=False)
    return train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx)
