# 📝 GAN Training Report

**Generated on:** 2025-09-02 20:25:25

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
- **Final Generator Loss:** 2.1091
- **Final Discriminator Loss:** 9.7833
- **Final Discriminator Accuracy:** 0.3627
- **Loss Difference (D-G):** 7.6742

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
| Mean | 0.5062 | 356.2405 |
| Std | 0.8951 | 268.4386 |
| Min | 0.0000 | 9.7833 |
| Max | 4.2086 | 829.6883 |

### Discriminator Accuracy Statistics

- **Mean Accuracy:** 0.1918
- **Standard Deviation:** 0.0842
- **Min Accuracy:** 0.0686
- **Max Accuracy:** 0.3725

## 💡 Training Insights

❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.

⚠️ **Discriminator Too Weak**: Low accuracy may indicate poor discriminator training.

⚠️ **Training Instability**: High variance in recent losses detected.

## 🎯 Model Performance Metrics

### Power-Voltage Distribution Analysis

- **JS Divergence (Power):** 0.934446
- **JS Divergence (Voltage):** 0.411679
- **KL Divergence (Power):** 4.167546
- **KL Divergence (Voltage):** 1.175730

### Edge Power Distribution Analysis

- **JS Divergence:** 0.917840
- **KL Divergence:** 3.905321

## 📈 Statistical Summary

### Node Features (Power & Voltage)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Power Mean | -0.0006 | -11.1009 | 11.1003 |
| Voltage Mean | 1.0238 | 1.0058 | 0.0180 |
| Power Std | 0.6293 | 38.1930 | 37.5637 |
| Voltage Std | 0.0149 | 0.0494 | 0.0345 |

### Edge Features (Power Flow)

| Metric | Real Data | Generated Data | Difference |
|--------|-----------|----------------|------------|
| Edge Power Mean | 0.3245 | 0.6139 | 0.2895 |
| Edge Power Std | 0.0091 | 0.2794 | 0.2703 |

## 🔍 Data Quality Assessment

### Sample Sizes

- **Real Data Points:** 108
- **Generated Data Points:** 108
- **Edge Data Points (Real):** 108
- **Edge Data Points (Generated):** 108

## 📋 Performance Interpretation

### Jensen-Shannon Divergence Analysis

❌ **Poor Performance**: High JS divergence values indicate significant distribution mismatch.

### Key Findings

- **Power Mean Deviation:** 1976244.00%
- **Voltage Mean Deviation:** 1.76%
- **Edge Power Mean Deviation:** 89.21%

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

### Model Status: Requires Improvement

- Consider the following improvements:
  - Review network architecture
  - Adjust learning rates
  - Increase training data diversity
  - Implement advanced GAN techniques (e.g., progressive growing, spectral normalization)

---

*This report was automatically generated from GAN training results. All metrics and visualizations are based on the latest model checkpoint.*
