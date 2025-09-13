# 📝 GAN Training Report

**Generated on:** 2025-09-13 16:52:57

## ⚙️ Configuration

```yaml
data:
  dis_scenario_type: 8
  gen_scenario_type: 9
  load_std: 0.1
  net_name: net_A
  noise: true
  num_samples: 8192
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
- **Final Generator Loss:** 0.7399
- **Final Discriminator Loss:** 6.3470
- **Final Discriminator Accuracy:** 0.7242
- **Loss Difference (D-G):** 5.6071

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
| Mean | 1.3660 | 35.3823 |
| Std | 1.3203 | 24.4697 |
| Min | 0.0476 | 5.5223 |
| Max | 11.0070 | 75.6415 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.5422
- **Standard Deviation:** 0.2945
- **Min Accuracy:** 0.0049
- **Max Accuracy:** 0.7435

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

✅ **Optimal Discriminator Performance**: Accuracy around 50% indicates good balance.

✅ **Stable Training**: Low variance in recent losses indicates stability.

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0014 | -0.0268 | 0.0254 |
| Voltage Mean | 1.0262 | 1.0301 | 0.0039 |
| Power Std | 0.5582 | 0.1007 | 0.4574 |
| Voltage Std | 0.0205 | 0.0052 | 0.0153 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3102 | 0.2347 | 0.0754 |
| Edge Power Std | 0.2520 | 0.1906 | 0.0615 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 2688
- **Generated Data Points:** 2688
- **Edge Data Points (Real):** 2688
- **Edge Data Points (Generated):** 2688

### Key Findings

- **Power Mean Deviation:** 1862.05%
- **Voltage Mean Deviation:** 0.38%
- **Edge Power Mean Deviation:** 24.33%

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.195296
- **JS Divergence (Voltage):** 0.800438
- **KL Divergence (Power):** 2.385324
- **KL Divergence (Voltage):** 6.271747

### Edge Power Distribution Analysis

- **JS Divergence:** 0.195179
- **KL Divergence:** 1.030922

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
