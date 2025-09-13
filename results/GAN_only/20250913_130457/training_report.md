# 📝 GAN Training Report

**Generated on:** 2025-09-13 13:05:00

## ⚙️ Configuration

```yaml
data:
  dis_scenario_type: 8
  gen_scenario_type: 9
  load_std: 0.1
  net_name: net_A
  noise: true
  num_samples: 400
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
  lr: 1.0e-06
  schedular_min_lr: 1.0e-05
  weight_decay: 0.0
training_G:
  lr: 0.0001
  schedular_min_lr: 1.0e-05
  weight_decay: 0.0
training_GAN:
  disc_iter: 2
  early_stopping: false
  feature_matching: true
  gen_iter: 1
  num_epoch: 30
  val_patience: 5
```

## 📊 Training Summary

- **Total Epochs:** 30
- **Final Generator Loss:** 1.4009
- **Final Discriminator Loss:** 702.4436
- **Final Discriminator Accuracy:** 0.2812
- **Loss Difference (D-G):** 701.0427

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
| Mean | 4.8390 | 772.6432 |
| Std | 10.0215 | 69.5598 |
| Min | 1.3704 | 702.4436 |
| Max | 54.7109 | 969.7886 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.2889
- **Standard Deviation:** 0.0092
- **Min Accuracy:** 0.2812
- **Max Accuracy:** 0.3234

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Discriminator Too Weak**: Low accuracy may indicate poor discriminator training.

⚠️ **Training Instability**: High variance in recent losses detected.

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0013 | -0.0213 | 0.0200 |
| Voltage Mean | 1.0259 | 1.0071 | 0.0188 |
| Power Std | 0.5645 | 0.6816 | 0.1171 |
| Voltage Std | 0.0205 | 0.0258 | 0.0053 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3130 | -0.6094 | 0.9223 |
| Edge Power Std | 0.2563 | 1.3298 | 1.0735 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 1680
- **Generated Data Points:** 1680
- **Edge Data Points (Real):** 1680
- **Edge Data Points (Generated):** 1680

### Key Findings

- **Power Mean Deviation:** 1515.97%
- **Voltage Mean Deviation:** 1.83%
- **Edge Power Mean Deviation:** 294.71%

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.475922
- **JS Divergence (Voltage):** 0.547767
- **KL Divergence (Power):** 2.111761
- **KL Divergence (Voltage):** 3.048230

### Edge Power Distribution Analysis

- **JS Divergence:** 0.532918
- **KL Divergence:** 1.648485

## 📊 Generated Visualizations

The following visualizations have been generated and saved:

1. **Power-Voltage KDE Comparison**
![pv_compare](FIG_GAN_power_voltage_kde_comparison.png)
2. **Edge Power KDE Distribution**
![edge_power](FIG_GAN_edge_power_kde_enhanced.png)
1. **Power-Voltage Histogram Comparison**
![pv_compare](FIG_ch_results_GAN_power_voltage_hist_comparison.png)
2. **Edge Power Histogram Distribution**
![edge_power](FIG_GAN_edge_power_hist_enhanced.png)
2. **P Real vs. Generated line plots**![rg_line](FIG_GAN_PVPedge_lineplot.png)
## Physical Consistency of Imputed Measurements1. Power Flow Converges? converged

2. **Simulated vs. Generated Voltage Profile**
![sim_gen_v](FIG_GAN_Simulated_vs_Generated_V.png)---

*This report was automatically generated from GAN training results. All metrics and visualizations are based on the latest model checkpoint.*
