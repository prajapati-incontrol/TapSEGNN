# 📝 Report 
 
## ⚙️ Configuration 

```yaml
data:
  load_std: 0.1
  net_name: MVO
  noise: true
  num_samples: 256
  scaler: true
  scenario_type: 9
  trafo_ids: []
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
  name: NEGATRegressor
  node_out_features: 32
  normalize: true
  trafo_hop: 5
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

- **Batchwise Average Test Loss**: `3.089635e-01`
- **RMSE_V**: `1.190721e-04`
- **RMSE_A**: `1.638120e+00`
- **MAE_V**: `9.313975e-05`
- **MAE_A**: `1.001190e+00`
- **MaxAE_V**: `5.707917e-04`
- **MaxAE_A**: `8.279173e+00`
- **NRMSE_V**: `1.084064e-05`
- **NRMSE_A**: `-3.050419e-02`

 Test Loss = 0.30896350741386414

### 📊 Predictions vs. Labels Bar Plot 

![Predictions vs Labels](va_barplot.png)

### Predictions vs. Labels Joint Distribution 

![Pred vs. Labels kde](va_pred_label_joint.png)