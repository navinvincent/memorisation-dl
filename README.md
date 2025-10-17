# Memorization in Deep Learning

Experiments on deep neural networks fitting arbitrary labellings of training data, exploring the fundamental question: **What factors govern learning vs. memorization in deep neural networks?**

The mechanisms underlying hypothesis selection in deep learning remain poorly understood. In classical machine learning theory, an overly large hypothesis space—such as that associated with overparameterized models—can fit datasets of arbitrary complexity, typically leading to overfitting to the training data and noise. Yet, in practice, deep neural networks routinely achieve state-of-the-art generalization across diverse benchmarks. Paradoxically, the same weakly regularized networks are also capable of perfectly memorizing random label assignments, indicating that stochastic optimization in conjunction with overparameterization can, in principle, memorize the entire training set.

![Hypothesis selection in deep learning?](figs/novak_et_al_implicit_reg.png "Novak et al. Implicit regularization conjecture")

*Source: [Novak et al. (2018)](https://openreview.net/forum?id=HJC2SzZCW)*

## Overview

This repository contains code to reproduce key experiments from [Zhang et al. (2017)](https://arxiv.org/abs/1611.03530), demonstrating that deep neural networks can fit arbitrary labellings of training data while exploring the role of explicit vs. implicit regularization.

### Key Experiments

1. **Label Corruption**: Train ResNet-18 on CIFAR-10 with varying ratios of corrupted labels (0% to 100%)
2. **Memorization Capacity**: Demonstrate that networks can achieve 100% training accuracy even with completely random labels
3. **Explicit Regularization**: Study the effect of weight decay, data augmentation, and batch normalization
4. **Implicit Regularization**: Explore learning dynamics without explicit regularization

## Installation

### Requirements

- Python 3.8+
- JAX (with GPU support recommended)
- PyTorch (for data loading only)
- Other dependencies listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd memorisation-dl

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
memorisation-dl/
├── configs/
│   └── default_config.py          # Default hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py              # Data loading and label corruption
│   │   └── transforms.py           # Custom data transforms
│   ├── models/
│   │   └── resnet.py               # ResNet architecture in JAX/Flax
│   ├── training/
│   │   └── trainer.py              # Training utilities and loops
│   └── utils/
│       ├── __init__.py
│       └── visualization.py        # Plotting utilities
├── experiments/
│   └── run_noise_experiments.py    # Script to run noise ratio experiments
├── figs/                           # Figures and visualizations
├── data/                           # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/                    # Model checkpoints
├── train.py                        # Main training script
└── README.md
```

## Usage

### Quick Start

Train a ResNet-18 on CIFAR-10 with 100% label corruption:

```bash
python train.py --noise_ratio 1.0 --max_epochs 100
```

### Basic Training

```bash
# Train with clean labels (no corruption)
python train.py --noise_ratio 0.0 --max_epochs 100

# Train with 50% label corruption
python train.py --noise_ratio 0.5 --max_epochs 100

# Enable data augmentation
python train.py --noise_ratio 0.0 --enable_augmentation

# Enable batch normalization
python train.py --noise_ratio 0.0 --enable_batch_norm

# Custom learning rate and weight decay
python train.py --lr 0.01 --weight_decay 1e-4
```

### Command Line Arguments

```
Data Arguments:
  --data_dir              Path to CIFAR-10 dataset (default: ./data)
  --checkpoints_dir       Directory to save checkpoints (default: ./checkpoints)

Training Arguments:
  --batch_size           Batch size for training (default: 128)
  --max_epochs           Maximum number of training epochs (default: 100)
  --seed                 Random seed (default: 42)

Model Arguments:
  --num_filters          Number of filters in first ResNet layer (default: 16)
  --enable_batch_norm    Enable batch normalization (flag)

Optimizer Arguments:
  --lr                   Initial learning rate (default: 0.1)
  --lr_decay             Learning rate decay factor (default: 0.95)
  --lr_step              Learning rate decay step (default: 1)
  --weight_decay         Weight decay coefficient (default: 0.0)

Data Augmentation and Noise:
  --noise_ratio          Ratio of labels to corrupt, 0.0 to 1.0 (default: 1.0)
  --enable_augmentation  Enable data augmentation (flag)

Other:
  --resume               Resume from latest checkpoint (flag)
  --plot_results         Plot training results (default: True)
```

### Running Experiments

#### Noise Ratio Sweep

Run experiments with different noise ratios (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):

```bash
python experiments/run_noise_experiments.py --output_dir ./experiments/noise_ratio
```

This will train 6 models with different corruption levels and save results to separate directories.

## Methodology

### Label Corruption

The `corrupt_labels` function implements label corruption by:

1. Randomly selecting a fraction of training samples (determined by `noise_ratio`)
2. Replacing each selected label with a uniformly random class **different** from the original
3. Ensuring reproducibility through fixed random seeds

Mathematically, for a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, we corrupt labels as:

$$\tilde{y}_i = \begin{cases} 
y_i & \text{with probability } 1 - p \\
\text{Uniform}(\{0, \ldots, C-1\} \setminus \{y_i\}) & \text{with probability } p
\end{cases}$$

where $p$ is the `noise_ratio` and $C$ is the number of classes.

### Loss Function

The training objective is cross-entropy loss with optional L2 weight decay:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \ell_{\text{CE}}(f_\theta(x_i), \tilde{y}_i) + \frac{\lambda}{2}\|\theta\|_2^2$$

where:
- $f_\theta$ is the neural network with parameters $\theta$
- $\ell_{\text{CE}}$ is the softmax cross-entropy loss
- $\lambda$ is the weight decay coefficient
- $\tilde{y}_i$ are the (possibly corrupted) labels

### Learning Rate Schedule

Following Zhang et al. (2017), we use exponential learning rate decay:

$$\eta_t = \eta_0 \cdot \gamma^{t/T}$$

where:
- $\eta_0$ is the initial learning rate (default: 0.1)
- $\gamma$ is the decay rate (default: 0.95)
- $T$ is the decay step in epochs (default: 1)
- $t$ is the current training step

### Stopping Criterion

Training continues until either:
1. Training accuracy reaches 99.9% (memorization achieved)
2. Maximum number of epochs is reached

## Key Results

### Memorization Capability

Deep networks can fit **arbitrary** labellings of the training data:
- With 100% label corruption, ResNet-18 can still achieve near 100% training accuracy
- Test accuracy drops to ~10% (random chance for 10 classes), confirming pure memorization

### Explicit vs Implicit Regularization

**Explicit Regularization** (weight decay, data augmentation, batch norm):
- Helps improve generalization on clean data
- Can slow down memorization but doesn't prevent it entirely
- Not sufficient to explain generalization in practice

**Implicit Regularization** (architecture, optimizer, initialization):
- Even without explicit regularization, networks often learn generalizable patterns
- SGD dynamics, architecture inductive biases play crucial roles
- Key open question: what implicit biases lead to generalization?

## Implementation Details

### Framework

- **JAX/Flax**: Neural network implementation and training
- **PyTorch**: Data loading (converted to NumPy for JAX)
- **Optax**: Optimizers and learning rate schedules

### Model Architecture

ResNet-18 adapted for CIFAR-10:
- 16 base filters (reduced from 64 for faster training)
- No pre-activation, standard residual connections
- Optional batch normalization
- Global average pooling before classification

### Training Configuration

Default settings (as per assignment):
- **No explicit regularization**: weight_decay=0, no augmentation, no batch norm
- **Batch size**: 128
- **Optimizer**: SGD with momentum 0.9
- **Learning rate**: 0.1 with exponential decay (0.95 per epoch)
- **Max epochs**: 100

## References

1. **[Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530)** - Zhang et al., ICLR 2017.
   - Main paper reproduced in this codebase
   
2. **[A Closer Look at Memorization in Deep Networks](https://icml.cc/Conferences/2017/ScheduleMultitrack?event=1327)** - Arpit et al., ICML 2017.
   - Studies memorization dynamics during training
   
3. **[Sensitivity and Generalization in Neural Networks: an Empirical Study](https://openreview.net/forum?id=HJC2SzZCW)** - Novak et al., ICLR 2018.
   - Proposes implicit regularization hypothesis
   
4. **[In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning](https://openreview.net/forum?id=6AzZb_7Qo0e)** - Neyshabur, Tomioka, and Srebro, ICLR Workshop 2015.
   - Early work on implicit regularization

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors with JAX:

```bash
# Limit GPU memory pre-allocation
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
```

Or modify the code to use CPU:
```python
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

### Slow Training

- Reduce `num_filters` (default: 16)
- Increase `batch_size` if memory allows
- Reduce `max_epochs`

## Acknowledgements

- Data loading utilities adapted from [JAX official documentation](https://jax.readthedocs.io/)
- ResNet implementation adapted from [Flax ImageNet example](https://github.com/google/flax/tree/master/examples/imagenet)
- PIL to NumPy transforms adapted from [JAX-ResNet-CIFAR10](https://github.com/hushon/JAX-ResNet-CIFAR10)

## License

This project is for educational purposes as part of a deep learning course assignment.


