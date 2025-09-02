# 📝 GAN Training Report

**Generated on:** 2025-09-02 20:29:49

## ⚙️ Configuration

```yaml
data:
  dis_scenario_type: 8
  gen_scenario_type: 9
  load_std: 0.1
  net_name: net_4bus
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
- **Final Generator Loss:** 0.8541
- **Final Discriminator Loss:** 1.0279
- **Final Discriminator Accuracy:** 0.7514
- **Loss Difference (D-G):** 0.1738

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
| Mean | 0.9958 | 19.7426 |
| Std | 0.9175 | 96.7855 |
| Min | 0.0000 | 0.6828 |
| Max | 9.3832 | 621.8201 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.8124
- **Standard Deviation:** 0.2256
- **Min Accuracy:** 0.2515
- **Max Accuracy:** 0.9997

## 💡 Training Insights

⚠️ **Moderate Convergence**: Some imbalance between generator and discriminator.

⚠️ **Discriminator Too Strong**: High accuracy may indicate generator struggling.

✅ **Stable Training**: Low variance in recent losses indicates stability.

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.470115
- **JS Divergence (Voltage):** 0.285197
- **KL Divergence (Power):** 8.954829
- **KL Divergence (Voltage):** 4.486367

### Edge Power Distribution Analysis

- **JS Divergence:** 0.467548
- **KL Divergence:** 1.256920

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0018 | 0.2968 | 0.2986 |
| Voltage Mean | 1.0263 | 1.0234 | 0.0029 |
| Power Std | 0.6179 | 0.2007 | 0.4172 |
| Voltage Std | 0.0184 | 0.0030 | 0.0154 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3249 | 0.3048 | 0.0200 |
| Edge Power Std | 0.0088 | 0.0312 | 0.0224 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 256
- **Generated Data Points:** 256
- **Edge Data Points (Real):** 256
- **Edge Data Points (Generated):** 256

## 📋 Performance Interpretation

### Jensen-Shannon Divergence Analysis

⚠️ **Moderate Performance**: JS divergence values suggest some distribution differences.

### Key Findings

- **Power Mean Deviation:** 16854.23%
- **Voltage Mean Deviation:** 0.28%
- **Edge Power Mean Deviation:** 6.17%

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

## 💡 Recommendations

### Model Status: Good Performance, Minor Improvements Possible

- The model performs well but could benefit from:
  - Extended training with more epochs
  - Hyperparameter tuning
  - Data augmentation techniques

---

*This report was automatically generated from GAN training results. All metrics and visualizations are based on the latest model checkpoint.*
