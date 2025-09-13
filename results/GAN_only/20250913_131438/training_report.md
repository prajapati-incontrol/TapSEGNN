# 📝 GAN Training Report

**Generated on:** 2025-09-13 13:14:41

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
- **Final Generator Loss:** 0.0020
- **Final Discriminator Loss:** 94.7537
- **Final Discriminator Accuracy:** 0.2468
- **Loss Difference (D-G):** 94.7516

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
| Mean | 0.0003 | 116.5810 |
| Std | 0.0005 | 11.6313 |
| Min | 0.0000 | 94.7537 |
| Max | 0.0020 | 137.4114 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.2465
- **Standard Deviation:** 0.0001
- **Min Accuracy:** 0.2465
- **Max Accuracy:** 0.2468

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Discriminator Too Weak**: Low accuracy may indicate poor discriminator training.

⚠️ **Training Instability**: High variance in recent losses detected.

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0011 | 0.2856 | 0.2867 |
| Voltage Mean | 1.0271 | 1.0206 | 0.0064 |
| Power Std | 0.5585 | 1.4952 | 0.9367 |
| Voltage Std | 0.0206 | 0.0358 | 0.0152 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3090 | -0.0008 | 0.3098 |
| Edge Power Std | 0.2515 | 0.4328 | 0.1813 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 2688
- **Generated Data Points:** 2688
- **Edge Data Points (Real):** 2688
- **Edge Data Points (Generated):** 2688

### Key Findings

- **Power Mean Deviation:** 26412.80%
- **Voltage Mean Deviation:** 0.63%
- **Edge Power Mean Deviation:** 100.27%

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.520620
- **JS Divergence (Voltage):** 0.496269
- **KL Divergence (Power):** 1.701220
- **KL Divergence (Voltage):** 1.498290

### Edge Power Distribution Analysis

- **JS Divergence:** 0.362821
- **KL Divergence:** 1.557304

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
