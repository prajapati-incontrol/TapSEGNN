# 📝 GAN Training Report

**Generated on:** 2025-09-12 11:47:56

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
  lr: 0.01
  schedular_min_lr: 1.0e-05
  weight_decay: 0.0
training_G:
  lr: 0.01
  schedular_min_lr: 1.0e-05
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
- **Final Generator Loss:** 15.1345
- **Final Discriminator Loss:** 130.2290
- **Final Discriminator Accuracy:** 1.0000
- **Loss Difference (D-G):** 115.0944

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
| Mean | 19.0938 | 126.8337 |
| Std | 16.2203 | 17.5567 |
| Min | 12.2680 | 27.5888 |
| Max | 102.4847 | 138.3703 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.9992
- **Standard Deviation:** 0.0058
- **Min Accuracy:** 0.9588
- **Max Accuracy:** 1.0000

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Discriminator Too Strong**: High accuracy may indicate generator struggling.

✅ **Stable Training**: Low variance in recent losses indicates stability.

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0016 | 8.0997 | 8.1013 |
| Voltage Mean | 1.0260 | 0.8710 | 0.1550 |
| Power Std | 0.5736 | 1.7589 | 1.1854 |
| Voltage Std | 0.0202 | 0.0385 | 0.0183 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3148 | 29.5661 | 29.2513 |
| Edge Power Std | 0.2631 | 33.5935 | 33.3304 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 2688
- **Generated Data Points:** 2688
- **Edge Data Points (Real):** 2688
- **Edge Data Points (Generated):** 2688

### Key Findings

- **Power Mean Deviation:** 511672.47%
- **Voltage Mean Deviation:** 15.11%
- **Edge Power Mean Deviation:** 9292.64%

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.999974
- **JS Divergence (Voltage):** 0.999964
- **KL Divergence (Power):** 22.777000
- **KL Divergence (Voltage):** 25.446111

### Edge Power Distribution Analysis

- **JS Divergence:** 0.883053
- **KL Divergence:** 3.771560

## 📊 Generated Visualizations

The following visualizations have been generated and saved:

1. **Power-Voltage KDE Comparison**
![pv_compare](FIG_GAN_power_voltage_kde_comparison.pdf)
2. **Edge Power KDE Distribution**
![edge_power](FIG_GAN_edge_power_kde_enhanced.png)
---

*This report was automatically generated from GAN training results. All metrics and visualizations are based on the latest model checkpoint.*
