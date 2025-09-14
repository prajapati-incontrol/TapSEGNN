# 📝 Report 
 
## ⚙️ Configuration 

```yaml
data:
  load_std: 0.1
  net_name: net_A
  noise: true
  num_samples: 256
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
  num_epochs: 30
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

- **Trafo 0 Accuracy**: `0.25925925925925924`
- **Trafo 1 Accuracy**: `0.0`
- **Trafo 2 Accuracy**: `0.3333333333333333`
- **Trafo 3 Accuracy**: `0.25925925925925924`
- **Trafo 4 Accuracy**: `0.2962962962962963`
- **Trafo 5 Accuracy**: `0.2962962962962963`
- **Trafo 6 Accuracy**: `0.25925925925925924`
- **Trafo 7 Accuracy**: `0.2222222222222222`
- **Trafo 8 Accuracy**: `0.18518518518518517`
- **Trafo 9 Accuracy**: `0.07407407407407407`
- **Trafo 10 Accuracy**: `0.0`
- **Trafo 11 Accuracy**: `0.2962962962962963`
- **Trafo 12 Accuracy**: `0.25925925925925924`
- **Trafo 13 Accuracy**: `0.2222222222222222`
- **Trafo 14 Accuracy**: `0.37037037037037035`
- **Trafo 15 Accuracy**: `0.4444444444444444`
- **Trafo 16 Accuracy**: `0.2962962962962963`
- **Trafo 17 Accuracy**: `0.1111111111111111`
- **Average_Accuracy_all_trafos**: `0.23251028806584362`
- **Batchwise Average Test Loss**: `4.366068e+00`
- **RMSE_V**: `7.545237e-04`
- **RMSE_A**: `1.168769e-02`
- **MAE_V**: `5.181559e-04`
- **MAE_A**: `5.809752e-03`
- **MaxAE_V**: `2.229408e-03`
- **MaxAE_A**: `1.120682e-01`
- **NRMSE_V**: `6.843123e-05`
- **NRMSE_A**: `1.240111e-03`

 Test Loss = 4.366068363189697

### 📊 Predictions vs. Labels Bar Plot 

![Predictions vs Labels](va_barplot.png)

### Predictions vs. Labels Joint Distribution 

![Pred vs. Labels kde](va_pred_label_joint.png)