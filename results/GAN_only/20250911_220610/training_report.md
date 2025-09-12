# 📝 GAN Training Report

**Generated on:** 2025-09-11 22:06:12

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
  disc_iter: 1
  early_stopping: false
  gen_iter: 5
  num_epoch: 50
  val_patience: 5
```

## 📊 Training Summary

- **Total Epochs:** 50
- **Final Generator Loss:** 15.0619
- **Final Discriminator Loss:** 0.2040
- **Final Discriminator Accuracy:** 1.0000
- **Loss Difference (D-G):** -14.8579

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
| Mean | 281.8253 | 41.6159 |
| Std | 1561.8971 | 221.3984 |
| Min | 0.0000 | 0.2002 |
| Max | 11090.2502 | 1579.7844 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.9171
- **Standard Deviation:** 0.1958
- **Min Accuracy:** 0.2669
- **Max Accuracy:** 1.0000

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Discriminator Too Strong**: High accuracy may indicate generator struggling.

⚠️ **Training Instability**: High variance in recent losses detected.

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.652057
- **JS Divergence (Voltage):** 0.956734
- **KL Divergence (Power):** 2.230170
- **KL Divergence (Voltage):** 5.477131

### Edge Power Distribution Analysis

- **JS Divergence:** 0.904216
- **KL Divergence:** 4.235827

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0014 | 2.0692 | 2.0705 |
| Voltage Mean | 1.0267 | 1.4860 | 0.4594 |
| Power Std | 0.5557 | 4.3431 | 3.7874 |
| Voltage Std | 0.0205 | 0.2473 | 0.2268 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3107 | 12.7719 | 12.4612 |
| Edge Power Std | 0.2515 | 11.5098 | 11.2583 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 2688
- **Generated Data Points:** 2688
- **Edge Data Points (Real):** 2688
- **Edge Data Points (Generated):** 2688

### Key Findings

- **Power Mean Deviation:** 151266.36%
- **Voltage Mean Deviation:** 44.74%
- **Edge Power Mean Deviation:** 4010.28%

## 📊 Generated Visualizations

The following visualizations have been generated and saved:

1. **Power-Voltage KDE Comparison**
   - File: `FIG_GAN_power_voltage_kde_comparison.pdf`
   - File: `FIG_ch_results_GAN_power_voltage_kde_comparison.png`
   - Shows joint distribution comparison of power and voltage values

2. **Edge Power KDE Distribution**
   - File: `FIG_GAN_edge_power_kde_enhanced.pdf`
   - File: `FIG_GAN_edge_power_kde_enhanced.png`
   - Shows distribution comparison of edge power flows

## 🔧 Technical Details

### Model Architecture

- **Model Type:** Graph Generative Adversarial Network (GAN)
- **Node Features:** Power (P) and Voltage Magnitude (|V|)
- **Edge Features:** Power Flow (P+)
- **Scaling:** StandardScaler applied to both node and edge features

### Evaluation Metrics

- **Jensen-Shannon Divergence:** Measures similarity between probability distributions (0 = identical, 1 = completely different)
- **Kullback-Leibler Divergence:** Measures information loss when approximating one distribution with another
- **Kernel Density Estimation:** Used for probability density estimation from samples

---

*This report was automatically generated from GAN training results. All metrics and visualizations are based on the latest model checkpoint.*
