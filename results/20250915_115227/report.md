# 📝 Report 
 
## ⚙️ Configuration 

```yaml
data:
  load_std: 0.1
  net_name: net_A
  noise: true
  num_samples: 4096
  scaler: true
  scenario_type: 9
  trafo_ids:
  - all
description: 'Tap-position prediction and state-estimation using Graph  Neural Networks

  '
device: cpu
loader:
  batch_size: 64
  split_list:
  - 0.8
  - 0.1
  - 0.1
model:
  bias: true
  edge_out_features: 64
  gat_head: 1
  gat_out_features: 32
  k_hop_edge: 1
  k_hop_node: 3
  list_edge_hidden_features:
  - 128
  list_node_hidden_features:
  - 64
  name: MultiTapSEGNN
  node_out_features: 32
  normalize: true
  trafo_hop: 1
plot:
  last_epochs: None
  plot_log: true
training:
  early_stopping: false
  loss_tap_weight: 0.1
  lr: 0.01
  num_epochs: 100
  schedular_min_lr: 0.0001
  val_patience: 5
  weight_decay: 0.001
```

## Load distribution to sample synthetic power flow results

![LoadP box](loadstd_box.png)

![LoadQ box](loadstdq_box.png)

## Power Flow Results Distribution

The plots below show the variability of all states sampled by adding standard deviation in load.

![V box](vmpu_box.png)

![A box](adeg_box.png)

![P box](pmw_box.png)

![Q box](qmvar_box.png)

## 📊 Label Distribution 
 
### Unscaled 

![Voltage Magnitude Labelunscale](vm_pu_label_unscaled.png)

![Voltage Angle Labelunscale](va_deg_label_unscaled.png)

### Scaled (Input to the model) 

![Voltage Magnitude Label](vm_pu_label.png)

![Voltage Angle Label](va_rad_label.png)

## 📊 Parameter Distribution 
 
![Line and Trafo Parameter Distribution](param_joint_dist.png)

## 📉 Loss curve 
 
![Training Loss](loss.png)

## 📉 Loss curve 
 
![Gradient Curve](gradient_norm.png)

## 🔎 Results 

- **Trafo 0 Accuracy**: `1.0`
- **Trafo 1 Accuracy**: `1.0`
- **Trafo 2 Accuracy**: `1.0`
- **Trafo 3 Accuracy**: `1.0`
- **Trafo 4 Accuracy**: `1.0`
- **Trafo 5 Accuracy**: `1.0`
- **Trafo 6 Accuracy**: `1.0`
- **Trafo 7 Accuracy**: `1.0`
- **Trafo 8 Accuracy**: `1.0`
- **Trafo 9 Accuracy**: `1.0`
- **Trafo 10 Accuracy**: `1.0`
- **Trafo 11 Accuracy**: `1.0`
- **Trafo 12 Accuracy**: `1.0`
- **Trafo 13 Accuracy**: `1.0`
- **Trafo 14 Accuracy**: `1.0`
- **Trafo 15 Accuracy**: `1.0`
- **Trafo 16 Accuracy**: `1.0`
- **Trafo 17 Accuracy**: `1.0`
- **Average_Accuracy_all_trafos**: `1.0`
- **Batchwise Average Test Loss**: `5.861463e-01`
- **RMSE_V**: `1.868687e-05`
- **RMSE_A**: `3.828440e-03`
- **MAE_V**: `1.267405e-05`
- **MAE_A**: `1.658980e-03`
- **MaxAE_V**: `1.306683e-04`
- **MaxAE_A**: `4.919270e-02`
- **NRMSE_V**: `1.694790e-06`
- **NRMSE_A**: `4.067122e-04`

 Test Loss = 0.5861462908131736

### 📊 Predictions vs. Labels Bar Plot 

![Predictions vs Labels](va_barplot.png)

### Predictions vs. Labels Joint Distribution 

![Pred vs. Labels kde](va_pred_label_joint.png)