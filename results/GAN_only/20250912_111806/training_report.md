# 📝 GAN Training Report

**Generated on:** 2025-09-12 11:18:07

## ⚙️ Configuration

```yaml
data:
  dis_scenario_type: 8
  gen_scenario_type: 9
  load_std: 0.1
  net_name: net_A
  noise: true
  num_samples: 4096
description: 'Configuration for GAN experiment in /notebooks/rq_4_gans.ipynb

  '
device: cpu
loader:
  batch_size: 64
model_D:
  hidden_channel: 64
  out_channel: 32
model_G:
  bias: true
  edge_out_features: 1
  gat_head: 1
  gat_out_features: 32
  k_hop_edge: 1
  k_hop_node: 3
  list_edge_hidden_features:
  - 128
  list_node_hidden_features:
  - 64
  name: NEGATGenerator
  node_out_features: 32
  normalize: true
training_D:
  lr: 1.0e-05
  schedular_min_lr: 1.0e-05
  weight_decay: 0.0
training_G:
  lr: 0.01
  schedular_min_lr: 0.0001
  weight_decay: 0.0
training_GAN:
  disc_iter: 2
  early_stopping: false
  feature_matching: true
  gen_iter: 1
  num_epoch: 50
  val_patience: 5
```

## 📊 Training Summary

- **Total Epochs:** 50
- **Final Generator Loss:** 20.8418
- **Final Discriminator Loss:** 169.5805
- **Final Discriminator Accuracy:** 0.7492
- **Loss Difference (D-G):** 148.7386

## 📈 Training Dynamics

The following plot shows the evolution of generator and discriminator losses along with discriminator accuracy throughout training:

![Training Dynamics](FIG_training_dynamics.png)

## 🎯 Convergence Analysis

The loss difference plot shows how close the GAN is to Nash equilibrium (where discriminator and generator losses are balanced):

![Loss Difference](FIG_loss_difference.png)

## 📋 Statistical Analysis

### Loss Statistics

| Metric | Generator | Discriminator |
|--------|-----------|---------------|
| Mean | 23.8043 | 353.6042 |
| Std | 76.5359 | 806.6219 |
| Min | 0.0000 | 59.3545 |
| Max | 541.2897 | 5365.6041 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.6261
- **Standard Deviation:** 0.1959
- **Min Accuracy:** 0.2492
- **Max Accuracy:** 0.7492

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Training Instability**: High variance in recent losses detected.

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

