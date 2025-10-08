import torch 
import torch.nn as nn 
from typing import  List, Union, Literal, Dict
from torch_geometric.data import Dataset 
from torch_geometric.loader import DataLoader 
import sys 
import os 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import copy 

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.model.graph_model import TAGNRegressor, FCNNRegressor, NERegressor, NRegressor
from src.model.graph_model import TapNRegressor, EdgeRegressor, EdgeLGRegressor, NEGATRegressor
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
                     num_nodes: int = None, 
                     output_pred_va: bool = False) -> Dict: 
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
    offset = 0.
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

    if output_pred_va: 
        return results, pred_se_va, label_se_va 
    else: 
        return results

##############################################################################################################################

def impute_dataset_with_model_G(model_G: nn.Module, 
                                dataset_w_mm: Dataset, 
                                sampled_input_data: Dict):     
    loader = DataLoader(dataset_w_mm, batch_size=1, shuffle=False)
    cp_sampled_input_data = copy.deepcopy(sampled_input_data)
    cp_sampled_input_data['node_input_feat_mm'] = torch.zeros_like(cp_sampled_input_data['node_input_feat'])
    cp_sampled_input_data['edge_input_feat_mm'] = torch.zeros_like(cp_sampled_input_data['edge_input_feat']) 

    # forward pass of model_G 
    for batch_id, batch in enumerate(loader): 
        with torch.no_grad(): 
            scaled_gen_pv, scaled_gen_p_edge = model_G(batch)
            # REPLACE (lazy implementation) the node and edge input features for TapSEGNN
            cp_sampled_input_data['node_input_feat_mm'][batch_id] = scaled_gen_pv
            cp_sampled_input_data['edge_input_feat_mm'][batch_id,:,0] = scaled_gen_p_edge 
            break
    return cp_sampled_input_data