# 📝 GAN Training Report

**Generated on:** 2025-09-13 13:10:41

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
- **Final Generator Loss:** 5.8067
- **Final Discriminator Loss:** 19.8369
- **Final Discriminator Accuracy:** 0.7594
- **Loss Difference (D-G):** 14.0302

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
| Mean | 14.8213 | 12.9618 |
| Std | 11.9666 | 5.2789 |
| Min | 5.8067 | 3.7130 |
| Max | 54.5136 | 19.8369 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.7594
- **Standard Deviation:** 0.0000
- **Min Accuracy:** 0.7594
- **Max Accuracy:** 0.7594

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Training Instability**: High variance in recent losses detected.

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0013 | 0.3309 | 0.3322 |
| Voltage Mean | 1.0242 | 1.0258 | 0.0016 |
| Power Std | 0.5584 | 0.7245 | 0.1661 |
| Voltage Std | 0.0201 | 0.0504 | 0.0302 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3095 | 0.6477 | 0.3382 |
| Edge Power Std | 0.2529 | 0.5449 | 0.2920 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 1680
- **Generated Data Points:** 1680
- **Edge Data Points (Real):** 1680
- **Edge Data Points (Generated):** 1680

### Key Findings

- **Power Mean Deviation:** 25316.72%
- **Voltage Mean Deviation:** 0.15%
- **Edge Power Mean Deviation:** 109.26%

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.449438
- **JS Divergence (Voltage):** 0.565479
- **KL Divergence (Power):** 2.045570
- **KL Divergence (Voltage):** 1.678816

### Edge Power Distribution Analysis

- **JS Divergence:** 0.425795
- **KL Divergence:** 1.383777

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
