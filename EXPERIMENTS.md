# Experiment Guide

This guide provides detailed instructions for running the experiments.

## Experiment 1: Label Corruption (Memorization Capacity)

This experiment demonstrates that deep networks can memorize arbitrary labellings.

### Quick Run
```bash
# Train with 100% label corruption
python train.py --noise_ratio 1.0 --max_epochs 100 --checkpoints_dir ./checkpoints/noise_1.0
```

### Expected Results
- **Training accuracy**: Should reach ~99.9% (network memorizes corrupted labels)
- **Test accuracy**: Should be ~10% (random chance for 10 classes)
- **Interpretation**: Network has sufficient capacity to memorize random labels

## Experiment 2: Noise Ratio Sweep

Compare memorization vs. generalization across different noise levels.

### Quick Run
```bash
python experiments/run_noise_experiments.py
```

### What it does
Trains 6 models with noise ratios: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

### Expected Results
- **noise_ratio=0.0**: High test accuracy (~85-90%)
- **noise_ratio=0.2-0.6**: Gradual degradation in test accuracy
- **noise_ratio=1.0**: Test accuracy at chance level (~10%)

### Analysis
```bash
python src/utils/analysis.py --experiment_dir ./experiments/noise_ratio --type noise
```

## Experiment 3: Explicit Regularization

Study the effect of different regularization techniques.

### Quick Run
```bash
python experiments/run_regularization_experiments.py
```

### What it does
Trains 5 models with different regularization:
1. **No regularization** (baseline)
2. **Weight decay** (L2 regularization)
3. **Data augmentation** (random crops, flips)
4. **Batch normalization**
5. **All combined**

### Expected Results
- Regularization slows down overfitting
- But doesn't prevent memorization entirely
- All variants can still fit corrupted labels if trained long enough

## Experiment 4: Custom Experiments

### Train with specific configuration
```bash
# Example: Train with 50% noise, data augmentation, and weight decay
python train.py \
    --noise_ratio 0.5 \
    --enable_augmentation \
    --weight_decay 1e-4 \
    --max_epochs 100 \
    --checkpoints_dir ./checkpoints/custom_experiment
```

### Resume from checkpoint
```bash
python train.py \
    --resume \
    --checkpoints_dir ./checkpoints/noise_1.0
```

## Understanding the Results

### Training Metrics

Each experiment saves:
- **Checkpoints**: Model weights at key epochs (every 10 epochs)
- **Training plot**: Loss, accuracy, and learning rate curves
- **Final statistics**: Printed to console

### Visualizations

Training plots show three panels:
1. **Loss curves**: Train vs. test cross-entropy loss
2. **Accuracy curves**: Train vs. test classification accuracy
3. **Learning rate schedule**: Exponential decay over time

### Key Observations

**Memorization (noise_ratio=1.0)**:
- Train loss → 0
- Train accuracy → 100%
- Test accuracy → 10% (chance)
- **Conclusion**: Network memorizes without learning patterns

**Generalization (noise_ratio=0.0)**:
- Train loss → low
- Train accuracy → ~95%
- Test accuracy → ~85-90%
- **Conclusion**: Network learns generalizable features

**Partial Noise (0 < noise_ratio < 1)**:
- Network fits both clean and corrupted labels
- Test accuracy degrades proportionally
- **Conclusion**: Network has capacity for both learning and memorization

## Interpreting the Experiments

### The Memorization Paradox

The key finding: **Same network, same optimizer, vastly different outcomes**

- With clean labels → generalization
- With random labels → memorization
- Same architecture, same optimization algorithm!

### What This Tells Us

1. **Capacity**: Overparameterized networks have huge representational capacity
2. **Optimization**: SGD can find solutions that fit arbitrary patterns
3. **Implicit Bias**: Something beyond explicit regularization guides learning
4. **Open Question**: What implicit biases lead to generalization?

### Theoretical Implications

Classical learning theory says:
- Large capacity → overfitting
- Regularization needed for generalization

Deep learning shows:
- Large capacity doesn't always → overfitting
- Regularization helps but isn't necessary
- **Something else** provides an inductive bias toward simple solutions

## Advanced Experiments

### Varying Model Capacity
```bash
# Smaller network (8 filters)
python train.py --num_filters 8 --noise_ratio 1.0

# Larger network (32 filters)
python train.py --num_filters 32 --noise_ratio 1.0
```

### Different Learning Rates
```bash
# Lower learning rate
python train.py --lr 0.01 --noise_ratio 1.0

# Higher learning rate
python train.py --lr 0.5 --noise_ratio 1.0
```

### With Batch Normalization
```bash
python train.py --enable_batch_norm --noise_ratio 1.0
```

## Troubleshooting

### Training is too slow
- Reduce `--num_filters` (default: 16)
- Reduce `--max_epochs`
- Use GPU if available (JAX auto-detects)

### Out of memory
- Reduce `--batch_size`
- Reduce `--num_filters`
- Use CPU: Set `JAX_PLATFORM_NAME=cpu`

### Poor results
- Check learning rate (default: 0.1)
- Verify data loading (should see "Downloading CIFAR-10")
- Check stopping criterion (trains until 99.9% accuracy)

## Further Reading

- **Zhang et al. (2017)**: "Understanding deep learning requires rethinking generalization"
- **Neyshabur et al. (2017)**: "Exploring generalization in deep learning"
- **Arpit et al. (2017)**: "A closer look at memorization in deep networks"

## Questions to Explore

1. Does the network memorize random labels differently than clean labels?
2. How does the learning trajectory differ between memorization and generalization?
3. Can we identify when the network starts memorizing vs. generalizing?
4. What role does the architecture play in implicit regularization?

These questions motivate ongoing research in deep learning theory!
