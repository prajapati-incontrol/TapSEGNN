# 📝 GAN Training Report

**Generated on:** 2025-09-13 13:06:17

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
- **Final Generator Loss:** 1.2978
- **Final Discriminator Loss:** 16.7689
- **Final Discriminator Accuracy:** 0.3141
- **Loss Difference (D-G):** 15.4712

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
| Mean | 37.6836 | 16.9664 |
| Std | 68.6693 | 2.1737 |
| Min | 1.2978 | 13.9669 |
| Max | 309.4356 | 22.8962 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.5367
- **Standard Deviation:** 0.1575
- **Min Accuracy:** 0.3141
- **Max Accuracy:** 0.7156

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

✅ **Optimal Discriminator Performance**: Accuracy around 50% indicates good balance.

⚠️ **Training Instability**: High variance in recent losses detected.

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0010 | 0.4663 | 0.4673 |
| Voltage Mean | 1.0261 | 1.0152 | 0.0109 |
| Power Std | 0.5654 | 1.0671 | 0.5017 |
| Voltage Std | 0.0206 | 0.0337 | 0.0131 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3116 | 0.4228 | 0.1112 |
| Edge Power Std | 0.2566 | 0.5588 | 0.3022 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 1680
- **Generated Data Points:** 1680
- **Edge Data Points (Real):** 1680
- **Edge Data Points (Generated):** 1680

### Key Findings

- **Power Mean Deviation:** 48658.22%
- **Voltage Mean Deviation:** 1.06%
- **Edge Power Mean Deviation:** 35.68%

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.440593
- **JS Divergence (Voltage):** 0.580597
- **KL Divergence (Power):** 1.550418
- **KL Divergence (Voltage):** 1.804007

### Edge Power Distribution Analysis

- **JS Divergence:** 0.304116
- **KL Divergence:** 0.761604

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
