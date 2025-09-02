# 📝 GAN Training Report

**Generated on:** 2025-09-02 20:17:39

## ⚙️ Configuration

```yaml
data:
  dis_scenario_type: 8
  gen_scenario_type: 9
  load_std: 0.1
  net_name: net_4bus
  noise: true
  num_samples: 256
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
  weight_decay: 1.0e-05
training_G:
  lr: 0.01
  schedular_min_lr: 0.0001
  weight_decay: 0.0
training_GAN:
  disc_iter: 2
  early_stopping: false
  gen_iter: 1
  num_epoch: 100
  val_patience: 5
```

## 📊 Training Summary

- **Total Epochs:** 100
- **Final Generator Loss:** 4.3214
- **Final Discriminator Loss:** 8.0331
- **Final Discriminator Accuracy:** 0.4681
- **Loss Difference (D-G):** 3.7117

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
| Mean | 2.4080 | 172.8184 |
| Std | 2.1217 | 338.8528 |
| Min | 0.0000 | 2.7856 |
| Max | 9.6424 | 2662.4357 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.4384
- **Standard Deviation:** 0.1307
- **Min Accuracy:** 0.2892
- **Max Accuracy:** 0.6458

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

✅ **Optimal Discriminator Performance**: Accuracy around 50% indicates good balance.

⚠️ **Training Instability**: High variance in recent losses detected.

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

