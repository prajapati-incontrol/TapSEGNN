# 📝 GAN Training Report

**Generated on:** 2025-09-13 17:05:55

## ⚙️ Configuration

```yaml
data:
  dis_scenario_type: 8
  gen_scenario_type: 9
  load_std: 0.1
  net_name: net_A
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
  num_epoch: 100
  val_patience: 5
```

## 📊 Training Summary

- **Total Epochs:** 100
- **Final Generator Loss:** 1.7576
- **Final Discriminator Loss:** 5.9236
- **Final Discriminator Accuracy:** 0.7549
- **Loss Difference (D-G):** 4.1660

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
| Mean | 30.8019 | 11.6998 |
| Std | 78.6616 | 5.4753 |
| Min | 1.6676 | 5.9236 |
| Max | 524.1329 | 27.0872 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.7570
- **Standard Deviation:** 0.0009
- **Min Accuracy:** 0.7549
- **Max Accuracy:** 0.7574

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Training Instability**: High variance in recent losses detected.

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0010 | -0.0679 | 0.0670 |
| Voltage Mean | 1.0266 | 1.0308 | 0.0042 |
| Power Std | 0.5480 | 0.7334 | 0.1854 |
| Voltage Std | 0.0210 | 0.0200 | 0.0010 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3072 | 0.1870 | 0.1203 |
| Edge Power Std | 0.2469 | 0.4248 | 0.1779 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 1134
- **Generated Data Points:** 1134
- **Edge Data Points (Real):** 1134
- **Edge Data Points (Generated):** 1134

### Key Findings

- **Power Mean Deviation:** 6800.14%
- **Voltage Mean Deviation:** 0.41%
- **Edge Power Mean Deviation:** 39.15%

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.438565
- **JS Divergence (Voltage):** 0.566630
- **KL Divergence (Power):** 1.604741
- **KL Divergence (Voltage):** 1.841794

### Edge Power Distribution Analysis

- **JS Divergence:** 0.396582
- **KL Divergence:** 1.155547

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
