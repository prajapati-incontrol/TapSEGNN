import torch 
import torch.nn as nn 
from typing import  List, Union, Literal, Dict
from torch_geometric.data import Dataset 
from torch_geometric.loader import DataLoader 
import sys 
import os 
import numpy as np 
from sklearn.preprocessing import StandardScaler

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.model.graph_model import TAGNRegressor, FCNNRegressor, NERegressor, TapGNN, NRegressor
from src.model.graph_model import TapNRegressor, EdgeRegressor, EdgeLGRegressor, NEGATRegressor, TapSEGNN
from src.model.graph_model import MultiTapSEGNN, NEGATRegressor, GATRegressor, NGATRegressor, NEGATRegressor_LGL
from utils.gen_utils import get_trafo_neighbors, get_rmse, get_mae, get_maxae, get_nrmse, precision_round
from utils.load_data_utils import inverse_scale, retrieve_trafo_minmaxedge
from utils.plot_utils import plot_two_vec
from src.training.trainer import eval_epoch_se, eval_epoch_multitapse, eval_epoch_fcnn_se

def initialize_model(model_name: str,
                      dataset: Dataset,
                      node_out_features: int,
                      list_node_hidden_features: List,
                      k_hop_node: int,
                      edge_out_features: int, 
                      list_edge_hidden_features: List,
                      k_hop_edge: int,
                      trafo_hop: int,
                      edge_index_list: List,
                      gat_out_features: int,
                      gat_head: int,
                      agg_op: str = "sum",  
                      bias: bool = True, 
                      normalize: bool =False, 
                      adj_norm: bool = True, 
                      device: torch.device = 'cpu',
                      ): 
    
    match model_name: # latest from top to bottom
        case 'NEGATRegressor_LGL': 
            model = NEGATRegressor_LGL(node_input_features=dataset[0][0].x.shape[1],
                                list_node_hidden_features=list_node_hidden_features,
                                node_out_features=node_out_features,
                                k_hop_node=k_hop_node,
                                edge_input_features=dataset[0][1].x.shape[1],
                                list_edge_hidden_features=list_edge_hidden_features,
                                edge_output_features=edge_out_features,
                                k_hop_edge=k_hop_edge,
                                gat_out_features=gat_out_features,
                                gat_head=gat_head,
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, 
                                device=device,
                                )
            
        case 'NGATRegressor':
            model = NGATRegressor(node_input_features=dataset[0][0].x.shape[1],
                                  list_node_hidden_features=list_node_hidden_features, 
                                  node_out_features=node_out_features, 
                                  k_hop_node=k_hop_node,
                                 gat_edge_features=dataset[0][1].x.shape[1], 
                                 gat_out_features=gat_out_features, 
                                 gat_head=gat_head, 
                                 bias=bias, 
                                 device=device,
                                 )
        
        case 'GATRegressor':
            model = GATRegressor(gat_in_features=dataset[0][0].x.shape[1],
                                 gat_edge_features=dataset[0][1].x.shape[1], 
                                 gat_out_features=gat_out_features, 
                                 gat_head=gat_head, 
                                 bias=bias, 
                                 device=device,
                                 )

        case 'MultiTapSEGNN':
            # returns numpy arrays
            tap_min, tap_max, trafo_edge = retrieve_trafo_minmaxedge(dataset[0][0])

            num_tap_classes = tap_max + 1 

            # trafo_neighbors = {trafo_id: {all neighbors}}
            trafo_neighbors = get_trafo_neighbors(edge_index=edge_index_list, 
                                                  trafo_edge=trafo_edge, 
                                                  trafo_hop=trafo_hop, 
                                                  case_multi=True)
            
            # num_trafo_neighbors = [num_nbor_trafo_0, num_nbor_trafo_1, etc..]
            num_trafo_neighbors = [len(trafo_neighbors[trafo_id]) for trafo_id in trafo_neighbors.keys()]


            model = MultiTapSEGNN(node_input_features=dataset[0][0].x.shape[1],
                                list_node_hidden_features=list_node_hidden_features,
                                node_out_features=node_out_features,
                                k_hop_node=k_hop_node,
                                edge_input_features=dataset[0][1].x.shape[1],
                                list_edge_hidden_features=list_edge_hidden_features,
                                edge_output_features=edge_out_features,
                                k_hop_edge=k_hop_edge,
                                trafo_hop=trafo_hop,
                                num_trafo_neighbors=num_trafo_neighbors, # for multi-tap this is a list.
                                trafo_out_features=num_tap_classes, # for multi-tap this is a list.
                                gat_out_features=gat_out_features,
                                gat_head=gat_head,
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, 
                                device=device,
                                )

        case 'TapSEGNN':

            tap_min, tap_max = dataset[0][0].y_trafo_label[1]
            
            # since only one trafo edge 
            trafo_edge = dataset[0][0].y_trafo_label[0]
            
            num_tap_classes = len(range(int(tap_min), int(tap_max)+1))
            num_trafo_neighbors = len(get_trafo_neighbors(edge_index_list, trafo_edge, trafo_hop))


            model = TapSEGNN(node_input_features=dataset[0][0].x.shape[1],
                                list_node_hidden_features=list_node_hidden_features,
                                node_out_features=node_out_features,
                                k_hop_node=k_hop_node,
                                edge_input_features=dataset[0][1].x.shape[1],
                                list_edge_hidden_features=list_edge_hidden_features,
                                edge_output_features=edge_out_features,
                                k_hop_edge=k_hop_edge,
                                trafo_hop=trafo_hop,
                                num_trafo_neighbors=num_trafo_neighbors, 
                                trafo_out_features=num_tap_classes,
                                gat_out_features=gat_out_features,
                                gat_head=gat_head,
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, 
                                device=device,
                                )
        case 'NEGATRegressor':
            model = NEGATRegressor(node_input_features=dataset[0][0].x.shape[1],
                                list_node_hidden_features=list_node_hidden_features,
                                node_out_features=node_out_features,
                                k_hop_node=k_hop_node,
                                edge_input_features=dataset[0][1].x.shape[1],
                                list_edge_hidden_features=list_edge_hidden_features,
                                edge_output_features=edge_out_features,
                                k_hop_edge=k_hop_edge,
                                gat_out_features=gat_out_features,
                                gat_head=gat_head,
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, 
                                device=device,
                                )
        case 'EdgeLGRegressor': 
            model = EdgeLGRegressor(edge_input_features=dataset[0][1].x.shape[1],
                                  list_edge_hidden_features=list_edge_hidden_features,
                                  edge_output_features=edge_out_features,
                                  k_hop_edge=k_hop_edge,
                                  bias=bias, 
                                  normalize=normalize,
                                  adj_norm=adj_norm, 
                                  device=device)
        case 'EdgeRegressor': 
            model = EdgeRegressor(edge_input_features=dataset[0][0].x.shape[1],
                                  list_edge_hidden_features=list_edge_hidden_features,
                                  edge_output_features=edge_out_features,
                                  k_hop_edge=k_hop_edge,
                                  bias=bias, 
                                  normalize=normalize,
                                  adj_norm=adj_norm, 
                                  device=device)
        case 'TapNRegressor':
            # get tap_min, tap_max 
            # print(dataset[0][0].y_trafo_label[1])

            tap_min, tap_max = dataset[0][0].y_trafo_label[1]
            
            # since only one trafo edge 
            trafo_edge = dataset[0][0].y_trafo_label[0]
            
            num_tap_classes = len(range(int(tap_min), int(tap_max)+1))
            num_trafo_neighbors = len(get_trafo_neighbors(edge_index_list, trafo_edge, trafo_hop))

            model = TapNRegressor(node_input_features=dataset[0][0].x.shape[1], # 4 
                                list_node_hidden_features=list_node_hidden_features,
                                node_out_features=dataset[0][0].y.shape[1], # 2
                                k_hop_node=k_hop_node,
                                trafo_hop=trafo_hop, 
                                num_trafo_neighbors = num_trafo_neighbors, 
                                trafo_out_features=num_tap_classes, 
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, 
                                device=device)
        case 'NRegressor': 
            model = NRegressor(node_input_features=dataset[0][0].x.shape[1],
                                list_node_hidden_features=list_node_hidden_features, 
                                node_out_features=node_out_features, 
                                k_hop_node=k_hop_node,  
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, # normalize the adjacency matrix (recommended)
                                device=device)
        case 'NERegressor':
            model = NERegressor(node_input_features=dataset[0][0].x.shape[1],
                                list_node_hidden_features=list_node_hidden_features,
                                node_out_features=node_out_features,
                                k_hop_node=k_hop_node,
                                edge_input_features=dataset[0][1].x.shape[1],
                                list_edge_hidden_features=list_edge_hidden_features,
                                edge_output_features=edge_out_features,
                                k_hop_edge=k_hop_edge,
                                agg_op=agg_op,
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, 
                                device=device,
                                )
        case 'TapGNN': 
            # get tap_min, tap_max 
            print(dataset[0][0].y_trafo_label[1])

            tap_min, tap_max = dataset[0][0].y_trafo_label[1]
            
            # since only one trafo edge 
            trafo_edge = dataset[0][0].y_trafo_label[0]
            
            num_tap_classes = len(range(int(tap_min), int(tap_max)+1))
            num_trafo_neighbors = len(get_trafo_neighbors(edge_index_list, trafo_edge, trafo_hop))

            model = TapGNN(node_input_features=dataset[0][0].x.shape[1],
                                list_node_hidden_features=list_node_hidden_features,
                                node_out_features=node_out_features,
                                k_hop_node=k_hop_node,
                                edge_input_features=dataset[0][1].x.shape[1],
                                list_edge_hidden_features=list_edge_hidden_features,
                                edge_output_features=edge_out_features,
                                k_hop_edge=k_hop_edge,
                                trafo_hop=trafo_hop,
                                num_trafo_neighbors=num_trafo_neighbors, 
                                trafo_out_features=num_tap_classes,
                                bias=bias, 
                                normalize=normalize, 
                                adj_norm=adj_norm, 
                                device=device,
                                )
        case 'TAGNRegressor4SE': 
            model = TAGNRegressor(node_in_features = dataset[0][0].x.shape[1], # 4
                                        node_hidden_features = list_node_hidden_features,
                                        node_out_features = dataset[0][0].y.shape[1], # 2
                                        k_hop_node = k_hop_node, 
                                        bias=bias, 
                                        normalize=normalize)
            model.name = model_name
        case 'TAGNRegressor4PF': 
            model = TAGNRegressor(node_in_features = dataset[0][0].x.shape[1], # 4
                                        node_hidden_features = list_node_hidden_features,
                                        node_out_features = dataset[0][0].x.shape[1], # 4
                                        k_hop_node = k_hop_node, 
                                        bias=bias, 
                                        normalize=normalize)
            model.name = model_name
        case 'FCNNRegressor': 
            model = FCNNRegressor(in_feat = dataset[0][0].x.shape[1], # 4
                                      hid_feat_list = list_node_hidden_features,
                                      out_feat = dataset[0][0].x.shape[1])
            model.name = model_name
        case _: 
            raise NameError("Invalid model name")
    
    return model

###########################################################################################################

def get_eval_results(test_loader: DataLoader, 
                     trained_model: nn.Module,
                     tap_weight: float,  
                     scaler: StandardScaler,
                     device: Literal['cpu','mps','cuda'] = 'cpu', 
                     fcnn: bool = False,
                     num_nodes: int = None) -> Dict: 
    """
    This function returns various performance metrics of the trained model. 
    Available metrics: 
    For regression 
            Test Loss per batch 
            RootMeanSquareError (RMSE) per graph 
            MeanAbsoluteError (MAE) per graph 
            MaxAbsoluteError (MaxAE) per graph 
            NormalizedRMSE (NRMSE) per graph 
    
    For classification: 
            Accuracy per batch 
    """
    offset = 10.
    batch = next(iter(test_loader))
    results = dict()
    if not fcnn: 
        num_graphs = np.float32(len(batch[0].ptr) - 1)
        # assert (trained_model.name == "NGATRegressor") | (trained_model.name == "GATRegressor") | (trained_model.name == "NEGATRegressor") | (trained_model.name == "MultiTapSEGNN"), "Eval only supports MultiTapSEGNN and NEGATRegressor"
        
        all_tap_acc = dict()
        

        criterion_se_v = nn.MSELoss()
        criterion_se_a = nn.L1Loss()

        with torch.no_grad():
            pred = trained_model(batch)
        if trained_model.name == "MultiTapSEGNN": 
            pred_se, pred_tap_logits = pred 
            for trafo_id in pred_tap_logits.keys(): 
                single_trafo_y_pred_tap = pred_tap_logits[trafo_id] # batch_size * num_tap_classes
                single_trafo_y_target_tap = batch[0].y_tap[:,trafo_id].to(device) # batch_size
                _, pred_tap = torch.max(single_trafo_y_pred_tap, dim=1) # batch_size
                # for each batch, correct predictions for all graphs / number of all graphs
                all_tap_acc[trafo_id] = float(sum(pred_tap == single_trafo_y_target_tap)) / float(num_graphs)
                results[f'Trafo {trafo_id} Accuracy'] = all_tap_acc[trafo_id]
            results['Average_Accuracy_all_trafos'] = sum(all_tap_acc.values())/len(pred_tap_logits.keys())
            test_loss, test_loss_se, test_loss_tap = eval_epoch_multitapse(trained_model, 
                                                                            test_loader, 
                                                                            weight=tap_weight,
                                                                            criterion_se_v=criterion_se_v, 
                                                                            criterion_se_a=criterion_se_a, 
                                                                            angle_weight=1.1, 
                                                                            device=device) 
        else: 
            test_loss = eval_epoch_se(trained_model, 
                                    test_loader, 
                                    criterion_se_v=criterion_se_v, 
                                    criterion_se_a=criterion_se_a, 
                                    angle_weight=1.1, ################################
                                    device=device) 
            pred_se = pred
        
        
        if scaler:
            print("Calculating results for StandardScaled Voltage and Angles.")
            pred_se_va = inverse_scale(pred_se, scaler=scaler) + offset 
            label_se_va = inverse_scale(batch[0].y, scaler=scaler) + offset
        else:     
            print("Calculating results for Voltage and Angles in pu and degree respectively")
            pred_se_va = pred_se + offset
            label_se_va = batch[0].y + offset
    
    else:
        with torch.no_grad(): 
            test_loss = eval_epoch_fcnn_se(trained_model, 
                                           test_loader, 
                                           device=device)
        num_graphs = batch[0].shape[0]
        assert num_nodes != None, "Specify the number of nodes for evaluating the FCNN!"
        inputs, labels = batch 
        pred_se_fcnn = trained_model(inputs)
        labels_se_reshaped = labels.reshape((num_graphs * num_nodes, 2))
        pred_se_reshaped = pred_se_fcnn.reshape(((num_graphs * num_nodes, 2)))
        label_se_va = inverse_scale(labels_se_reshaped, scaler=scaler) + offset 
        pred_se_va = inverse_scale(pred_se_reshaped, scaler=scaler) + offset 
        

    results['Batchwise Average Test Loss'] = "{:e}".format(test_loss)
    results['RMSE_V'] = "{:e}".format(get_rmse(pred_se_va[:,0], label_se_va[:,0].to(device)) / num_graphs) 
    results['RMSE_A'] = "{:e}".format(get_rmse(pred_se_va[:,1], label_se_va[:,1].to(device)) / num_graphs)

    results['MAE_V'] = "{:e}".format(get_mae(pred_se_va[:,0], label_se_va[:,0].to(device)) / num_graphs) 
    results['MAE_A'] = "{:e}".format(get_mae(pred_se_va[:,1], label_se_va[:,1].to(device)) / num_graphs)

    results['MaxAE_V'] = "{:e}".format(get_maxae(pred_se_va[:,0], label_se_va[:,0].to(device)) / num_graphs) 
    results['MaxAE_A'] = "{:e}".format(get_maxae(pred_se_va[:,1], label_se_va[:,1].to(device)) / num_graphs)

    results['NRMSE_V'] = "{:e}".format(get_nrmse(pred_se_va[:,0], label_se_va[:,0].to(device)) / num_graphs) 
    results['NRMSE_A'] = "{:e}".format(get_nrmse(pred_se_va[:,1], label_se_va[:,1].to(device)) / num_graphs)

    return results 

###########################################################################################################


# def evaluate_se_model(model: nn.Module,
#                    test_loader: DataLoader, 
#                    sampled_input_data: dict,
#                    get_scaled_rmse: bool = True,
#                    device: Literal['cpu','mps','cuda'] = 'cpu'): 
#     """This method evaluates the model by calculating RMSE for voltage and angle per graph. 
#     Moreover, if the model accounts for tap-position, then it also calculates the accuracy ber batch in the loader."""

#     print(f"The test loader as {len(test_loader)} batch/es. \n")

#     # RMSE for all batches
#     rmse_v_all_batches = []
#     rmse_a_all_batches = []

#     with torch.no_grad(): 
#         for _, batch in enumerate(test_loader):
#             num_graphs = np.float32(len(batch[0].ptr)-1)
#             pred_se = model(batch)
            
#             if get_scaled_rmse:
#                 # get RMSE of unscaled (original) voltage and angles with labels
#                 rmse_se_v = get_rmse(pred_se[:,0], batch[0].y[:,0])/ num_graphs
#                 rmse_se_a = get_rmse(pred_se[:,1], batch[0].y[:,1])/ num_graphs # for each graph
#                 rmse_v_all_batches.append(rmse_se_v)
#                 rmse_a_all_batches.append(rmse_se_a)

#             else: 
#                 # predicted voltage inverse-scaled
#                 pred_se_unscaled = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])

#                 # labels inverse-scaled 
#                 y_unscaled = inverse_scale(batch[0].y, scaler=sampled_input_data['scaler_y_label'])

#                 # get RMSE of unscaled (original) voltage and angles with labels
#                 rmse_se_v = get_rmse(pred_se_unscaled[:,0], y_unscaled[:,0])/ num_graphs
#                 rmse_se_a = get_rmse(pred_se_unscaled[:,1], y_unscaled[:,1])/ num_graphs # for each graph
#                 rmse_v_all_batches.append(rmse_se_v)
#                 rmse_a_all_batches.append(rmse_se_a)
        

#     # RMSE Voltage
#     rmse_v_values = np.array(rmse_v_all_batches)
#     batch_max_rmse_v = np.max(rmse_v_values)
#     batch_min_rmse_v = np.min(rmse_v_values)
#     batch_max_rmse_v_index = int(np.argmax(rmse_v_values))
#     batch_min_rmse_v_index = int(np.argmin(rmse_v_values))

#     # RMSE Angle
#     rmse_a_values = np.array(rmse_a_all_batches)
#     batch_max_rmse_a = np.max(rmse_a_values)
#     batch_min_rmse_a = np.min(rmse_a_values)
#     batch_max_rmse_a_index = int(np.argmax(rmse_a_values))
#     batch_min_rmse_a_index = int(np.argmin(rmse_a_values))

#     # Final prints
#     print(f"Batch-max RMSE Voltage = {batch_max_rmse_v} at batch {batch_max_rmse_v_index} with scaling = {get_scaled_rmse}\n")
#     print(f"Batch-min RMSE Voltage = {batch_min_rmse_v} at batch {batch_min_rmse_v_index} with scaling = {get_scaled_rmse}\n")
#     print(f"Batch-max RMSE Angle = {batch_max_rmse_a} at batch {batch_max_rmse_a_index} with scaling = {get_scaled_rmse}\n")
#     print(f"Batch-min RMSE Angle = {batch_min_rmse_a} at batch {batch_min_rmse_a_index} with scaling = {get_scaled_rmse}\n")

#     return batch_max_rmse_v, batch_min_rmse_v, batch_max_rmse_a, batch_min_rmse_a

# def evaluate_tapse_model(model: nn.Module,
#                    test_loader: DataLoader, 
#                    sampled_input_data: dict, 
#                    device: Literal['cpu','mps','cuda'] = 'cpu'): 
#     """This method evaluates the model by calculating RMSE for voltage and angle per graph. 
#     Moreover, if the model accounts for tap-position, then it also calculates the accuracy ber batch in the loader."""
#     print(f"Using device {device} for evaluate_tapse_model...")
#     print(f"The test loader as {len(test_loader)} batches. \n")

#     # RMSE for all batches
#     rmse_v_all_batches = []
#     rmse_a_all_batches = []

#     # accuracy per batch 
#     batch_tap_accuracy = []

#     with torch.no_grad(): 
#         for _, batch in enumerate(test_loader):
#             num_graphs = np.float32(len(batch[0].ptr)-1)
#             # SE prediction and tap prediction logits 
#             pred_se, pred_tap_logits = model(batch)
#             _, pred_tap = torch.max(pred_tap_logits, dim=1)
#             batch_tap_accuracy.append(float(sum(pred_tap == batch[0].y_tap.to(device))) / np.float32(num_graphs))

#             # predicted voltage inverse-scaled
#             pred_se_unscaled = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])

#             # labels inverse-scaled 
#             y_unscaled = inverse_scale(batch[0].y, scaler=sampled_input_data['scaler_y_label'])

#             # get RMSE of unscaled (original) voltage and angles with labels
#             rmse_se_v = get_rmse(pred_se_unscaled[:,0], y_unscaled[:,0])/ num_graphs
#             rmse_se_a = get_rmse(pred_se_unscaled[:,1], y_unscaled[:,1])/ num_graphs # for each graph
#             rmse_v_all_batches.append(rmse_se_v)
#             rmse_a_all_batches.append(rmse_se_a)

#     # Accuracy
#     batch_max_accuracy = np.max(batch_tap_accuracy)
#     batch_min_accuracy = np.min(batch_tap_accuracy)
#     print(f"Batch-max accuracy = {batch_max_accuracy} \nBatch-min accuracy = {batch_min_accuracy} \n")

#     # RMSE Voltage
#     rmse_v_values = np.array(rmse_v_all_batches)
#     batch_max_rmse_v = np.max(rmse_v_values)
#     batch_min_rmse_v = np.min(rmse_v_values)
#     batch_max_rmse_v_index = int(np.argmax(rmse_v_values))
#     batch_min_rmse_v_index = int(np.argmin(rmse_v_values))

#     # RMSE Angle
#     rmse_a_values = np.array(rmse_a_all_batches)
#     batch_max_rmse_a = np.max(rmse_a_values)
#     batch_min_rmse_a = np.min(rmse_a_values)
#     batch_max_rmse_a_index = int(np.argmax(rmse_a_values))
#     batch_min_rmse_a_index = int(np.argmin(rmse_a_values))

#     # Final prints
#     print(f"Batch-max RMSE Voltage = {batch_max_rmse_v} at batch {batch_max_rmse_v_index}\n")
#     print(f"Batch-min RMSE Voltage = {batch_min_rmse_v} at batch {batch_min_rmse_v_index}\n")
#     print(f"Batch-max RMSE Angle = {batch_max_rmse_a} at batch {batch_max_rmse_a_index} \n")
#     print(f"Batch-min RMSE Angle = {batch_min_rmse_a} at batch {batch_min_rmse_a_index} \n")

#     return batch_min_accuracy, batch_max_accuracy, batch_max_rmse_v, batch_min_rmse_v, batch_max_rmse_a, batch_min_rmse_a
        
# def evaluate_multitapse_model(model: nn.Module,
#                    test_loader: DataLoader, 
#                    sampled_input_data: dict, 
#                    device: Literal['cpu','mps','cuda'] = 'cpu'): 
#     """This method evaluates the model by calculating RMSE for voltage and angle per graph. 
#     Moreover, if the model accounts for tap-position, then it also calculates the accuracy ber batch in the loader."""
    
#     print(f"The test loader as {len(test_loader)} batches. \n")

#     # RMSE for all batches
#     rmse_v_all_batches = []
#     rmse_a_all_batches = []
    
#     # to initialize the batch_all_tap_accuracy
#     batch_all_tap_accuracy = {trafo_id: [] for trafo_id in range(len(sampled_input_data['y_trafo_label'][0]))}
    
#     with torch.no_grad(): 
#         for batch_id, batch in enumerate(test_loader):
#             num_graphs = np.float32(len(batch[0].ptr)-1)
#             print(f"Evaluating batch {batch_id} in testloader, having {num_graphs} graphs")
#             pred_se, pred_tap_logits = model(batch)
#             # print(pred_se, batch[0].y)
#             for trafo_id in pred_tap_logits.keys(): 
#                 single_trafo_y_pred_tap = pred_tap_logits[trafo_id] # batch_size * num_tap_classes
#                 single_trafo_y_target_tap = batch[0].y_tap[:,trafo_id].to(device) # batch_size
#                 _, pred_tap = torch.max(single_trafo_y_pred_tap, dim=1) # batch_size

#                 # for each batch, correct predictions for all graphs / number of all graphs
#                 batch_all_tap_accuracy[trafo_id].append(float(sum(pred_tap == single_trafo_y_target_tap)) / float(num_graphs))

#                 # predicted voltage inverse-scaled
#                 pred_se_unscaled = inverse_scale(pred_se.cpu(), scaler=sampled_input_data['scaler_y_label'])

#                 # labels inverse-scaled 
#                 y_unscaled = inverse_scale(batch[0].y, scaler=sampled_input_data['scaler_y_label'])

#                 # get RMSE of unscaled (original) voltage and angles with labels
#                 rmse_se_v = get_rmse(pred_se_unscaled[:,0], y_unscaled[:,0])/ num_graphs
#                 rmse_se_a = get_rmse(pred_se_unscaled[:,1], y_unscaled[:,1])/ num_graphs # for each graph
#                 rmse_v_all_batches.append(rmse_se_v)
#                 rmse_a_all_batches.append(rmse_se_a)

#     # Print the worst accuracy for each transformer 
#     best_acc_all_tap_all_batch = {trafo_id: min(batch_all_tap_accuracy[trafo_id]) for trafo_id in range(len(sampled_input_data['y_trafo_label'][0]))}
#     worst_acc_all_tap_all_batch = {trafo_id: max(batch_all_tap_accuracy[trafo_id]) for trafo_id in range(len(sampled_input_data['y_trafo_label'][0]))}
    
#     sorted_best_acc = sorted(best_acc_all_tap_all_batch.items(), key=lambda x: x[1],reverse=True) # descending max 
#     sorted_worst_acc = sorted(best_acc_all_tap_all_batch.items(), key=lambda x: x[1]) # ascending min 

#     # Average Model Accuracy: All taps in batch / all graphs in batch 

#     print("First 5 transformers with highest accuracy: \n")
#     first5 = []
#     for k, v in sorted_best_acc[:5]:
#         print(F"Trafo ID: {k}, Accuracy: {v}")
#         first5.append(v)

#     print("Last 5 transformers with lowest accuracy: \n")
#     last5 = []
#     for k, v in sorted_worst_acc[:5]:
#         print(F"Trafo ID: {k}, Accuracy: {v}")
#         last5.append(v)
    
#     # RMSE Voltage
#     rmse_v_values = np.array(rmse_v_all_batches)
#     batch_max_rmse_v = np.max(rmse_v_values)
#     batch_min_rmse_v = np.min(rmse_v_values)
#     batch_max_rmse_v_index = int(np.argmax(rmse_v_values))
#     batch_min_rmse_v_index = int(np.argmin(rmse_v_values))

#     # RMSE Angle
#     rmse_a_values = np.array(rmse_a_all_batches)
#     batch_max_rmse_a = np.max(rmse_a_values)
#     batch_min_rmse_a = np.min(rmse_a_values)
#     batch_max_rmse_a_index = int(np.argmax(rmse_a_values))
#     batch_min_rmse_a_index = int(np.argmin(rmse_a_values))

#     # Final prints
#     print(f"Batch-max RMSE Voltage = {batch_max_rmse_v} at batch {batch_max_rmse_v_index}\n")
#     print(f"Batch-min RMSE Voltage = {batch_min_rmse_v} at batch {batch_min_rmse_v_index}\n")
#     print(f"Batch-max RMSE Angle = {batch_max_rmse_a} at batch {batch_max_rmse_a_index} \n")
#     print(f"Batch-min RMSE Angle = {batch_min_rmse_a} at batch {batch_min_rmse_a_index} \n")
        
#     return first5, last5, batch_max_rmse_v, batch_min_rmse_v, batch_max_rmse_a, batch_min_rmse_a

# def evaluate_edger_model(model: nn.Module,
#                    test_loader: DataLoader, 
#                    sampled_input_data: dict): 
#     """This method evaluates the model by calculating RMSE for voltage and angle per graph. 
#     Moreover, if the model accounts for tap-position, then it also calculates the accuracy ber batch in the loader."""
#     with torch.no_grad(): 
#             batch = next(iter(test_loader))
#             num_graphs = np.float32(len(batch[0].ptr)-1)
#             pred_e = model(batch) 


#     rmse_feature = get_rmse(pred_e[:,0], batch[0].y[:,0])/num_graphs # since y is same in hodge-data and LG data 
#     print(f"RMSE of feature 0 for {model.name} = {rmse_feature}")
#     rmse_feature1 = get_rmse(pred_e[:,1], batch[0].y[:,1])/num_graphs # since y is same in hodge-data and LG data 
#     print(f"RMSE of feature 1 for {model.name} = {rmse_feature1}")
#     rmse_feature2 = get_rmse(pred_e[:,2], batch[0].y[:,2])/num_graphs # since y is same in hodge-data and LG data 
#     print(f"RMSE of feature 2 for {model.name} = {rmse_feature2}")
        
#     return rmse_feature, rmse_feature1, rmse_feature2






# def evaluate_model(model: nn.Module,
#                    test_loader: DataLoader, 
#                    sampled_input_data: dict): 
#     """This method evaluates the model by calculating RMSE for voltage and angle per graph. 
#     Moreover, if the model accounts for tap-position, then it also calculates the accuracy ber batch in the loader."""

#     tapse_models = {"MultiTapSEGNN","TapSEGNN","TapGNN","TapNRegressor","NERegressor", "NRegressor", "TAGNRegressor4SE","NEGATRegressor"}
#     edger_models = {"EdgeRegressor", "EdgeLGRegressor"}
    
#     print(f"The test loader as {len(test_loader)} batches. \n")

#     if model.name in tapse_models:
#         with torch.no_grad(): 
#             # RMSE for all batches
#             rmse_v_all_batches = []
#             rmse_a_all_batches = []

#             # accuracy per batch 
#             batch_tap_accuracy = []

#             for _, batch in enumerate(test_loader):
#                 num_graphs = np.float32(len(batch[0].ptr)-1)

#                 if model.name in {"TapGNN","TapNRegressor","TapSEGNN"}:
#                     # SE prediction and tap prediction logits 
#                     pred_se, pred_tap_logits = model(batch)
#                     _, pred_tap = torch.max(pred_tap_logits, dim=1)
#                     batch_tap_accuracy.append(np.float32(sum(pred_tap == batch[0].y_tap)) / np.float32(num_graphs))
#                     # Accuracy
#                     batch_max_accuracy = np.max(batch_tap_accuracy)
#                     batch_min_accuracy = np.min(batch_tap_accuracy)
#                     print(f"Batch-max accuracy = {batch_max_accuracy} \nBatch-min accuracy = {batch_min_accuracy} \n")
#                 elif model.name in {"MultiTapSEGNN"}: 
#                     pred_se, pred_tap_logits = model(batch)
#                     batch_tap_accuracy = {trafo_id: [] for trafo_id in pred_tap_logits.keys()}
#                     for trafo_id in pred_tap_logits.keys(): 
#                         single_trafo_y_pred_tap = pred_tap_logits[trafo_id]
#                         single_trafo_y_target_tap = batch[0].y_tap[:,trafo_id]
#                         _, pred_tap = torch.max(single_trafo_y_pred_tap, dim=1)
#                         print(pred_tap, single_trafo_y_target_tap)
#                         batch_tap_accuracy[trafo_id].append(float(sum(pred_tap == single_trafo_y_target_tap)) / float(num_graphs))
#                     sorted_batch_tap_accuracy = sorted(batch_tap_accuracy.items(), key=lambda x: x[1][0])
#                     print("First 5 transformers with highest accuracy: \n")
#                     first5 = []
#                     for k, v in sorted_batch_tap_accuracy[:5]:
#                         print(F"Trafo ID: {k}, Accuracy: {v[0]}")
#                         first5.append(v[0])

#                     print("Last 5 transformers with lowest accuracy: \n")
#                     last5 = []
#                     for k, v in sorted_batch_tap_accuracy[-5:]:
#                         print(F"Trafo ID: {k}, Accuracy: {v[0]}")
#                         last5.append(v[0])
#                 else: 
                    

#                 # predicted voltage inverse-scaled
#                 pred_se_unscaled = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])

#                 # labels inverse-scaled 
#                 y_unscaled = inverse_scale(batch[0].y, scaler=sampled_input_data['scaler_y_label'])

#                 # get RMSE of unscaled (original) voltage and angles with labels
#                 rmse_se_v = get_rmse(pred_se_unscaled[:,0], y_unscaled[:,0])/ num_graphs
#                 rmse_se_a = get_rmse(pred_se_unscaled[:,1], y_unscaled[:,1])/ num_graphs # for each graph
#                 rmse_v_all_batches.append(rmse_se_v)
#                 rmse_a_all_batches.append(rmse_se_a)

#             # RMSE Voltage
#             rmse_v_values = np.array(rmse_v_all_batches)
#             batch_max_rmse_v = np.max(rmse_v_values)
#             batch_min_rmse_v = np.min(rmse_v_values)
#             batch_max_rmse_v_index = int(np.argmax(rmse_v_values))
#             batch_min_rmse_v_index = int(np.argmin(rmse_v_values))

#             # RMSE Angle
#             rmse_a_values = np.array(rmse_a_all_batches)
#             batch_max_rmse_a = np.max(rmse_a_values)
#             batch_min_rmse_a = np.min(rmse_a_values)
#             batch_max_rmse_a_index = int(np.argmax(rmse_a_values))
#             batch_min_rmse_a_index = int(np.argmin(rmse_a_values))

#             # Final prints
#             print(f"Batch-max RMSE Voltage = {batch_max_rmse_v} at batch {batch_max_rmse_v_index}\n")
#             print(f"Batch-min RMSE Voltage = {batch_min_rmse_v} at batch {batch_min_rmse_v_index}\n")
#             print(f"Batch-max RMSE Angle = {batch_max_rmse_a} at batch {batch_max_rmse_a_index} \n")
#             print(f"Batch-min RMSE Angle = {batch_min_rmse_a} at batch {batch_min_rmse_a_index} \n")
        

#         if model.name in {"TapGNN","TapNRegressor","TapSEGNN"}:
#             return batch_min_accuracy, batch_max_accuracy, batch_max_rmse_v, batch_min_rmse_v, batch_max_rmse_a, batch_min_rmse_a
#         elif model.name in {"MultiTapSEGNN"}:
#             return first5, last5, batch_max_rmse_v, batch_min_rmse_v, batch_max_rmse_a, batch_min_rmse_a
#         else: 
#             return batch_max_rmse_v, batch_min_rmse_v, batch_max_rmse_a, batch_min_rmse_a
        

#                    # # Plotting the voltage and angle for a batch with maximum RMSE V 
#             # max_rmse_v_arg = int(np.argmax(np.array(rmse_v_all_batches)))
#             # max_rmse_a_arg = int(np.argmax(np.array(rmse_a_all_batches)))

#             # for i, batch in enumerate(test_loader):
#             #     if i == max_rmse_v_arg:
#             #         pred_se = model(batch)
#             #         # predicted voltage inverse-scaled
#             #         worst_se_pred_v = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])

#             #         # labels inverse-scaled 
#             #         worst_y_label_v = inverse_scale(batch[0].y, scaler=sampled_input_data['scaler_y_label'])
#             #     if i == max_rmse_a_arg: 
#             #         pred_se = model(batch)

#             #         worst_se_pred_a = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])
#             #         worst_y_label_a = inverse_scale(batch[0].y, scaler=sampled_input_data['scaler_y_label'])
#             #     if model.name in {"TapGNN","TapNRegressor","TapSEGNN"}:
#             #         min_acc_tap_pos = np.argmin(batch_tap_accuracy)


#             # if model.name in {"TapGNN","TapNRegressor","TapSEGNN"}:
#             #     min_acc_tap_pos = np.argmin(batch_tap_accuracy)
#             #     for i, batch in enumerate(test_loader)
#     elif model.name in edger_models: 
#         with torch.no_grad(): 
#                 batch = next(iter(test_loader))
#                 num_graphs = np.float32(len(batch[0].ptr)-1)
#                 pred_e = model(batch) 
#                 rmse_feature = get_rmse(pred_e[:,0], batch[0].y[:,0])/num_graphs # since y is same in hodge-data and LG data 
#                 print(f"RMSE of feature 0 for {model.name} = {rmse_feature}")
#                 rmse_feature1 = get_rmse(pred_e[:,1], batch[0].y[:,1])/num_graphs # since y is same in hodge-data and LG data 
#                 print(f"RMSE of feature 1 for {model.name} = {rmse_feature1}")
#                 rmse_feature2 = get_rmse(pred_e[:,2], batch[0].y[:,2])/num_graphs # since y is same in hodge-data and LG data 
#                 print(f"RMSE of feature 2 for {model.name} = {rmse_feature2}")
            
#         return rmse_feature, rmse_feature1, rmse_feature2
    
#     else: 
#         raise NotImplementedError

def save_model(trained_model: nn.Module, 
               path: str):
    torch.save(trained_model.state_dict(), path)