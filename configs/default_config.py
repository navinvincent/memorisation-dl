"""
Default configuration for memorization experiments
"""

# Data configuration
CIFAR10_MEAN = (0.4919, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
NUM_CLASSES = 10

# Training hyperparameters
SEED = 42
ENABLE_AUGMENTATION = False
ENABLE_BATCH_NORM = False
BATCH_SIZE = 128
NOISE_RATIO = 1.0  # float between 0. and 1.
WEIGHT_DECAY = 0.0
LR = 0.1
LR_DECAY = 0.95
LR_STEP = 1
MAX_EPOCHS = 100

# Paths
DATA_DIR = "./data"
CHECKPOINTS_DIR = "./checkpoints"

# Model configuration
MODEL_NAME = "resnet18"
NUM_FILTERS = 16  # Reduced for faster training on CIFAR-10
