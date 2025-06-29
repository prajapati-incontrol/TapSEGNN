import networkx as nx 
import numpy as np
import pandapower as pp
import warnings 
import torch
import copy
import os
import pickle
import pandas as pd 
from typing import Tuple, List, Dict
from torch_geometric.data import Data, Batch, Dataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pylab
import time 
import joblib 
from sklearn.preprocessing import StandardScaler 

# warnings as errors to trace the source of error 
np.seterr(all='raise')


from utils.gen_utils import get_array_mask, tensor_any_nan, get_edge_index_from_ppnet, scale_numeric_columns, inverse_scale
from utils.ppnet_utils import add_branch_parameters, get_positive_power_flow, get_power_flow_edge_index

#########################################################################################################

def load_sampled_input_data(sc_type: int, 
                            net: pp.pandapowerNet, 
                            num_samples: int,
                            p_std: float = 1e-2,
                            noise: float = 0.0,
                            trafo_ids: List = [4],
                            std_meas: float = 0.0, 
                            std_pseudo: float = 0.0, 
                            scaler: bool = True,
                            p_true: float = 0.5, # for GANs
) -> Dict:
    """
    Load the data for the given scenario. 
    
    Args: 
        sc_type (int): Scenario type 
        net (pp.pandapowerNet): Pandapower Network 
        num_samples (int): Number of samples to collect 
        noise (float): Gaussian noise standard deviation to perturb the power flow results 
        trafo_ids (List): List of transformer index to perform prediction on. If all then trafo_ids = ['all']

    Returns: 
        sampled_input_data (dict): Dictionary of sampled input data, with keys varying for each scenario. 
    """
    
    sampled_input_data = dict()
    edge_index = get_edge_index_from_ppnet(net=net)
    sampled_input_data['edge_index'] = edge_index

    match sc_type: 
            case 1: 
                """
                Node Features: V, THETA, P, Q + Noise
                Edge Features: r, x, b, g, shift, incorrect_tap 
                Y Labels: V, THETA 
                Y Trafo Label: {sample_number: [(hv_bus, lv_bus), (tap_min, tap_max), correct_tap_pos]}
                

                """
                if len(trafo_ids) != 1:
                    raise ValueError("Scenario 1,2,3 only supports single trafo_id")
                
                node_input_feat, edge_input_feat, y_label, y_trafo_label = load_sc_1(net=net,
                                                                                     num_permutations=num_samples,
                                                                                     p_std=p_std,
                                                                                     trafo_id=trafo_ids,
                                                                                     noise=noise)

                if scaler:
                    print("Scaling inputs...")
                    node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                    edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat, categorical_cols=[5])
                    y_label, scaler_y = scale_numeric_columns(y_label)   

                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = scaler_n
                    sampled_input_data['scaler_edge'] = scaler_e
                    sampled_input_data['scaler_y_label'] = scaler_y
                else: 
                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = None
                    sampled_input_data['scaler_edge'] = None
                    sampled_input_data['scaler_y_label'] = None
                
                sampled_input_data['node_input_feat'] = node_input_feat
                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label'] = y_label
                sampled_input_data['y_trafo_label'] = y_trafo_label

                
                return sampled_input_data

            case 2: 
                """
                Node Features: V, P, Q + Noise (No THETA)
                Edge Features: r, x, b, g, shift, incorrect_tap 
                Y Labels: V, THETA 
                Y Trafo Label: {sample_number: [(hv_bus, lv_bus), (tap_min, tap_max), correct_tap_pos]}
                
                Available for Models:
                
                """

                if len(trafo_ids) != 1:
                    raise ValueError("Scenario 1,2,3 only supports single trafo_id")
                
                node_input_feat, edge_input_feat, y_label, y_trafo_label = load_sc_1(net=net,
                                                                                        num_permutations=num_samples,
                                                                                        p_std=p_std,
                                                                                        trafo_id=trafo_ids,
                                                                                        noise=noise)
                
                if scaler:
                    print("Scaling inputs...")
                    node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                    edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat, categorical_cols=[5])
                    y_label, scaler_y = scale_numeric_columns(y_label)   

                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = scaler_n
                    sampled_input_data['scaler_edge'] = scaler_e
                    sampled_input_data['scaler_y_label'] = scaler_y
                else: 
                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = None
                    sampled_input_data['scaler_edge'] = None
                    sampled_input_data['scaler_y_label'] = None

                # node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                # edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat, categorical_cols=[5])
                # y_label, scaler_y = scale_numeric_columns(y_label)

                # for only node data 
                vpq = torch.zeros((num_samples, node_input_feat.shape[1], 3))
                vpq[:, :, 0] = node_input_feat[:, :, 0]
                vpq[:, :, 1] = node_input_feat[:, :, 2]
                vpq[:, :, 2] = node_input_feat[:, :, 3]
                sampled_input_data['node_input_feat'] = vpq

                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label'] = y_label
                sampled_input_data['y_trafo_label'] = y_trafo_label
                # sampled_input_data['scaler_node'] = scaler_n
                # sampled_input_data['scaler_edge'] = scaler_e
                # sampled_input_data['scaler_y_label'] = scaler_y



                return sampled_input_data
            
            case 3: 
                """
                Node Features: V, P + Noise (No THETA and Q)
                Edge Features: r, x, b, g, shift, incorrect_tap 
                Y Labels: V, THETA 
                Y Trafo Label: {sample_number: [(hv_bus, lv_bus), (tap_min, tap_max), correct_tap_pos]}
                
                """

                if len(trafo_ids) != 1:
                    raise ValueError("Scenario 1,2,3 only supports single trafo_id")
                
                node_input_feat, edge_input_feat, y_label, y_trafo_label = load_sc_1(net=net,
                                                                                        num_permutations=num_samples,
                                                                                        p_std=p_std, 
                                                                                        trafo_id=trafo_ids,
                                                                                        noise=noise)

                if scaler:
                    print("Scaling inputs...")
                    node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                    edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat, categorical_cols=[5])
                    y_label, scaler_y = scale_numeric_columns(y_label)   

                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = scaler_n
                    sampled_input_data['scaler_edge'] = scaler_e
                    sampled_input_data['scaler_y_label'] = scaler_y
                else: 
                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = None
                    sampled_input_data['scaler_edge'] = None
                    sampled_input_data['scaler_y_label'] = None

                # for only node data 
                vpq = torch.zeros((num_samples, node_input_feat.shape[1], 2))
                vpq[:, :, 0] = node_input_feat[:, :, 0]
                vpq[:, :, 1] = node_input_feat[:, :, 2]
                sampled_input_data['node_input_feat'] = vpq
                
                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label'] = y_label
                sampled_input_data['y_trafo_label'] = y_trafo_label


                return sampled_input_data
            
            case 4: 
                """
                Edge Features: Random 6 features 
                Edge-Output: 3 features analytical function of input edge features. 

                This scenario is used to compare the model performance of Edge-Regression with Hodge-Laplacian vs. Linegraph Laplacian.  
                
                Available for Models:
                
                """

                edge_input_feat, y_label_edge = load_sc_4(net=net,
                                                num_permutations=num_samples)
                

                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label_edge'] = y_label_edge

                return sampled_input_data
            
            case 5: # to check if the net-data is causing NERegressor to not work.
                """
                Node Features: Random 4 features 
                Edge Features: 2 features analytical function of node-features 
                Y Label (node): 2 features analytical function of node-features

                This scenario is used to check the performance of node-regression, incorporating multiple edge features,
                tasks without any trafo prediction. 
                
                Available for Models: 
                
                """ 
                print(f"Using random node-edge-y input data.")
                node_input_feat, edge_attr, y = load_sc_5(net=net,
                                            num_permutations=num_samples, 
                                            edge_index=edge_index,
                                            )
                sampled_input_data['node_input_feat'] = node_input_feat
                sampled_input_data['edge_input_feat'] = edge_attr
                sampled_input_data['y_label'] = y
                sampled_input_data['y_trafo_label'] = None

                return sampled_input_data

            case 6: 
                """
                Node Features: V, P + Gaussian Noise 
                Edge Features: P_IN, Q_IN, P_OUT, Q_OUT, r, x, b, g, tap_pos (incorrect tap pos)
                Y Label (Node): V, THETA 
                Y Trafo Labels: {
                                    sample_number: 
                                    [               {'hv_node': 15,
                                                    'lv_node': 24,
                                                    'tap_min': -2.0,
                                                    'tap_max': 2.0,
                                                    'tap_pos': -1.0}, # correct tap position                                     
                                    }
                Node Mask: Using available measurements for Delft Technopolis as 1 else 0. Size is same as Node Features. 
                For other networks, randomly get the mask based on the sparsity probability. 

                Edge Mask: Using available measurements for Delft Technopolis as 1 else 0. Size is same as Edge Features.
                For other networks, randomly get the mask based on the sparsity probability.

                """
                node_input_feat, node_vapql, edge_input_feat, y_label, y_trafo_label, node_mask, edge_mask = load_sc_6(net=net, 
                                                                                                 num_samples=num_samples, 
                                                                                                 trafo_ids=trafo_ids, 
                                                                                                 noise=noise, 
                                                                                                )
                
                if scaler:
                    print("Scaling inputs...")
                    node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                    edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat, categorical_cols=[8])
                    y_label, scaler_y = scale_numeric_columns(y_label)   

                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = scaler_n
                    sampled_input_data['scaler_edge'] = scaler_e
                    sampled_input_data['scaler_y_label'] = scaler_y
                else: 
                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = None
                    sampled_input_data['scaler_edge'] = None
                    sampled_input_data['scaler_y_label'] = None


                # for only node data 
                sampled_input_data['node_input_feat'] = node_input_feat
                
                # store noise-less power flow data 
                sampled_input_data['res_bus_vm_pu'] = node_vapql[:,:,0].squeeze().T
                sampled_input_data['res_bus_va_deg'] = node_vapql[:,:,1].squeeze().T
                sampled_input_data['res_bus_p_mw'] = node_vapql[:,:,2].squeeze().T
                sampled_input_data['res_bus_q_mvar'] =  node_vapql[:,:,3].squeeze().T
                sampled_input_data['sim_load_p_mw'] = node_vapql[:,:,4].squeeze().T
                sampled_input_data['sim_load_q_mvar'] = node_vapql[:,:,5].squeeze().T
                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label'] = y_label 
                sampled_input_data['y_trafo_label'] = y_trafo_label
                sampled_input_data['node_mask'] = node_mask 
                sampled_input_data['edge_mask'] = edge_mask
                sampled_input_data['edge_index_dir'] = None

                nnz_node_mask = torch.count_nonzero(node_mask[0,:,:])
                nnz_edge_mask = torch.count_nonzero(edge_mask[0,:,:])

                print(f"Number of V, P measurements {nnz_node_mask} out of {node_mask[0,:,:].numel()}\n")
                print(f"Number of P_to, Q_to, P_from, Q_from measurements {nnz_edge_mask} out of {edge_mask[0,:,:].numel()}\n")

                return sampled_input_data
        
            case 7: 
                """
                Node Features: V, P + Gaussian Noise 
                Edge Features: P_IN, Q_IN, P_OUT, Q_OUT, r, x, b, g, shift, tap_pos (incorrect tap pos)
                Y Label (Node): V, THETA 
                Y Trafo Labels: {
                                    sample_number: 
                                    [               {'hv_node': 15,
                                                    'lv_node': 24,
                                                    'tap_min': -2.0,
                                                    'tap_max': 2.0,
                                                    'tap_pos': -1.0}, # correct tap position                                     
                                    }
                Available measurments have low standard deviation (std_meas) and other variables are perturbed with higher standard deviation 
                to simulate a pseudo-measurement (std_pseudo)
                """

                node_input_feat, edge_input_feat, y_label, y_trafo_label = load_sc_7(net=net, 
                                                                                        num_samples=num_samples, 
                                                                                        trafo_ids=trafo_ids, 
                                                                                        std_meas=std_meas, 
                                                                                        std_pseudo=std_pseudo, 
                                                                                        load_p_std=p_std)
                
                if scaler:
                    print("Scaling inputs...")
                    node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                    edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat, categorical_cols=[5])
                    y_label, scaler_y = scale_numeric_columns(y_label)   

                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = scaler_n
                    sampled_input_data['scaler_edge'] = scaler_e
                    sampled_input_data['scaler_y_label'] = scaler_y
                else: 
                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = None
                    sampled_input_data['scaler_edge'] = None
                    sampled_input_data['scaler_y_label'] = None

                # for only node data 
                sampled_input_data['node_input_feat'] = node_input_feat
                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label'] = y_label 
                sampled_input_data['y_trafo_label'] = y_trafo_label



                return sampled_input_data
        
            case 8: 
                """
                Real training data for discriminator. 
                Node Features: V, P + Gaussian Noise (if noise) with probability p_true, else Random Node Features
                Edge Features: 1, P^{+} + Gaussian Noise 
                Y label: 1 if node features sampled from V, P else 0 
                
                """

                node_input_feat, edge_input_feat, y_label = load_sc_8_discriminator(net=net, 
                                                                   num_samples=num_samples,
                                                                   p_true=p_true, 
                                                                   noise=noise, 
                                                                   load_p_std=p_std)
                
                if scaler:
                    print("Scaling inputs...")
                    node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                    edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat)

                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = scaler_n
                    sampled_input_data['scaler_edge'] = scaler_e
                else: 
                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = None
                    sampled_input_data['scaler_edge'] = None

                sampled_input_data['node_input_feat'] = node_input_feat
                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label'] = y_label 

                return sampled_input_data
                
            case 9: 
                """
                Node Features: V, P + Gaussian Noise 
                Edge Features: P_+ve, r, x, b, g, tap_pos (incorrect tap pos)
                Edge Dictionary: {(from_node, to_node),...,} # used to make directed edges in simplicial complex representation
                Y Label (Node): V, THETA 
                Y Trafo Labels: {
                                    sample_number: 
                                    [               {'hv_node': 15,
                                                    'lv_node': 24,
                                                    'tap_min': -2.0,
                                                    'tap_max': 2.0,
                                                    'tap_pos': -1.0}, # correct tap position                                     
                                    }
                Node Mask: Using available measurements for Delft Technopolis as 1 else 0. Size is same as Node Features. 
                For other networks, randomly get the mask based on the sparsity probability. 
                Edge Mask: Using available measurements for Delft Technopolis as 1 else 0. Size is same as Edge Features.
                For other networks, randomly get the mask based on the sparsity probability.
                """

                node_input_feat, node_vapql, edge_input_feat, y_label, y_trafo_label, node_mask, edge_mask, edge_index_dir = load_sc_9(net=net, 
                                                                                                 num_samples=num_samples, 
                                                                                                 trafo_ids=trafo_ids, 
                                                                                                 noise=noise, 
                                                                                                )
                
                if scaler:
                    print("Scaling inputs...")
                    node_input_feat, scaler_n = scale_numeric_columns(node_input_feat)
                    edge_input_feat, scaler_e = scale_numeric_columns(edge_input_feat, categorical_cols=[5])
                    y_label, scaler_y = scale_numeric_columns(y_label)   

                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = scaler_n
                    sampled_input_data['scaler_edge'] = scaler_e
                    sampled_input_data['scaler_y_label'] = scaler_y
                else: 
                    # used for inverse scaling for evaluating results 
                    sampled_input_data['scaler_node'] = None
                    sampled_input_data['scaler_edge'] = None
                    sampled_input_data['scaler_y_label'] = None


                # for only node data 
                sampled_input_data['node_input_feat'] = node_input_feat
                
                # store noise-less power flow data 
                sampled_input_data['res_bus_vm_pu'] = node_vapql[:,:,0].squeeze().T
                sampled_input_data['res_bus_va_deg'] = node_vapql[:,:,1].squeeze().T
                sampled_input_data['res_bus_p_mw'] = node_vapql[:,:,2].squeeze().T
                sampled_input_data['res_bus_q_mvar'] =  node_vapql[:,:,3].squeeze().T
                sampled_input_data['sim_load_p_mw'] = node_vapql[:,:,4].squeeze().T
                sampled_input_data['sim_load_q_mvar'] = node_vapql[:,:,5].squeeze().T
                sampled_input_data['edge_input_feat'] = edge_input_feat
                sampled_input_data['y_label'] = y_label 
                sampled_input_data['y_trafo_label'] = y_trafo_label
                sampled_input_data['node_mask'] = node_mask 
                sampled_input_data['edge_mask'] = edge_mask
                sampled_input_data['edge_index_dir'] = edge_index_dir

                nnz_node_mask = torch.count_nonzero(node_mask[0,:,:])
                nnz_edge_mask = torch.count_nonzero(edge_mask[0,:,:])

                print(f"Number of V, P measurements {nnz_node_mask} out of {node_mask[0,:,:].numel()}\n")
                print(f"Number of P_to, Q_to, P_from, Q_from measurements {nnz_edge_mask} out of {edge_mask[0,:,:].numel()}\n")

                return sampled_input_data
                
            case _: 
                raise NameError("Invalid scenario type!")


#####################################################################################

def load_sc_9(net: pp.pandapowerNet, 
              num_samples: int, 
              trafo_ids: List,
              noise: bool = True) -> Tuple: 
    """
    This data is considering the real measurement infrastructure for Delft Technopolis only. 
    """
    # if (len(trafo_ids) == 1) & (trafo_ids[0] != "all"): 
    #     raise ValueError("Total Transformer IDs should be more than 1, otherwise use scenario 1/2/3.")

    # check if trafo ids are all MV/LV Trafos 
    # remove switches 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    # node input features  
    num_buses = len(net.bus.index)
    num_node_features = 2 # V and P
    # node input features 
    node_input_features = np.zeros((num_samples, num_buses, num_node_features))
    node_vapq = np.zeros((num_samples, num_buses, 6)) # all V, A, P, Q, P_load, Q_load

    # edge input features 
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edges = num_lines + num_trafos
    num_edge_features = 6 # p_+ve, r, x, b, g, tap    

    # edge input features 
    edge_input_features = np.zeros((num_samples, num_edges, num_edge_features))

    # initialize edge_index_dir dictionary that captures the direction of edges with respect to active power flow measurement 
    edge_index_dir = dict() 

    # node output label features 
    y_label = np.zeros((num_samples, num_buses, 2)) # v and theta 

    # add r, x, b, g, shift, tap to net 
    net = add_branch_parameters(net)

    # variables to permutate in the net 
    pload_ref = copy.deepcopy(net.load['p_mw'].values)

    # defective trafo-tap positions 
    # check if the trafo ids exist in net df  
    if trafo_ids != ['all']:
        if len(net.trafo.loc[trafo_ids,:]) == len(trafo_ids):
                if net.trafo.loc[trafo_ids,"name"].str.contains("HV", na=False).any():
                    num_hv_trafos = sum(net.trafo.loc[trafo_ids,"name"].str.contains("HV", na=False)) 
                    nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
                    mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)
                    raise KeyError(f"{num_hv_trafos} Trafos selected are HV. Select out of following indices {mvlv_trafo_ids}\n")
    else: 
        print(f"Selecting all the MV/LV transformers in the network \n")
        nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
        trafo_ids = net.trafo.loc[nonhv_trafo_ids].index


    # mask the trafos so that 
    mask = net.trafo.index.isin(trafo_ids)

    # apply the mask and get the min-max trafo positions 
    tap_min, tap_max = list(net.trafo.loc[mask, "tap_min"]), list(net.trafo.loc[mask, "tap_max"])
    hv_bus_trafos, lv_bus_trafos = list(net.trafo.loc[mask, "hv_bus"]), list(net.trafo.loc[mask, "lv_bus"])

    # trafo tap labels For example: 
    # {sample_id: [{'hv_node': 10,
    #               'lv_node': 5,
    #                'tap_min: 0, # min-max should be non-negative to allow nn.CrossEntropy() to work
    #                'tap_max':4,
    #                'tap_pos':2}, # this will contain tap_pos based on which the pf is solved.
    #  {...}, {...}, ..., {...}]}

    # initialize the trafo label dictionary 
    y_trafo_label = dict()
    
    ### Adding data to the node and edge input feature matrices ###

    # since edge-features for r,x,b,g are not considered to change, assign them here. 
    # Vectorized edge tensor assignment

    # lines
    edge_input_features[:,:num_lines,1:5] = np.array(net.line[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32) 
    # trafos
    edge_input_features[:, num_lines:,1:5] = np.array(net.trafo[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32)
    
    # maximum retry: since power flow results cannot converge sometimes 
    max_retries = 20 
    seed_perm = 0 

    for iperm in range(num_samples):

        # set random tap position for edge_input_features tap_position 
        random_tap_for_edge_input = np.random.randint(tap_min, tap_max)
        net.trafo.loc[mask,"tap_pos"] = random_tap_for_edge_input.astype("int32")

        edge_input_features[iperm, num_lines:, 5] = np.array(net.trafo['tap_pos'].values)

        # now set random tap position for power flow calculations 
        random_tap_for_pfr = np.random.randint(tap_min, tap_max)

        net.trafo.loc[mask,"tap_pos"] = random_tap_for_pfr.astype("int32")

        # assign labels of trafo 
        y_trafo_label[iperm] = [
            {
                "hv_node": hv,
                "lv_node": lv,
                "tap_min": tmin+tmax,
                "tap_max": tmax+tmax,
                "tap_pos": int(tpos)+tmax
            }
            for hv, lv, tmin, tmax, tpos in zip(hv_bus_trafos, lv_bus_trafos, tap_min, tap_max, random_tap_for_pfr)
        ]

        retries = 0 
        while retries < max_retries:
            seed_perm += 1 
            try: 
                rng = np.random.default_rng(seed_perm)
                # permutate variables 
                pload = rng.normal(pload_ref, net.load.p_std.values)
                # qload = rng.normal(qload_ref, load_p_std)

                # modify the net data 
                net.load['p_mw'] = pload 
                # net.load['q_mvar'] = qload 
                node_vapq[iperm, :, 4][net.load.bus.values] = net.load.p_mw.values
                node_vapq[iperm, :, 5][net.load.bus.values] = net.load.q_mvar.values 

                net['converged'] = False 
                pp.runpp(net, max_iteration=50)

                # store the results for v and p only 
                node_input_features[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                node_vapq[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                y_label[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                node_input_features[iperm,:,1] = np.array(net.res_bus.p_mw.values)
                node_vapq[iperm,:,1] = np.array(net.res_bus.va_degree.values)
                y_label[iperm,:,1] = np.array(net.res_bus.va_degree.values)
                node_vapq[iperm,:,2] = np.array(net.res_bus.p_mw.values)
                node_vapq[iperm,:,3] = np.array(net.res_bus.q_mvar.values)

                # p_+ve measurements 
                edge_input_features[iperm, :, 0] = get_positive_power_flow(net)

                # edge index directed 
                edge_index_dir[iperm] = get_power_flow_edge_index(net)
                
                break # exit while loop if successful (reaching this line.)

            
            except Exception as e: 
                print(f"\t Error at permutation {iperm}: {e}")
                print(f"\t Retry #{retries} at {iperm} with a new random seed...")
                retries += 1
                continue

        if retries == max_retries:
            print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
            node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure

    # calculate the mask for available measurements at node and edge input features 
    # account for real measurements 
    node_mask = np.zeros((num_samples, num_buses, num_node_features))
    edge_mask = np.zeros((num_samples, num_edges, num_edge_features))
    
    
    if net.name == 'DFT_TNP':
        node_mask, edge_mask = get_dft_tnp_mask(net, node_mask, edge_mask)

    else: 
        sparsity_prob = 0.5
        node_mask = get_array_mask(node_input_features, sparsity_prob=sparsity_prob)
        edge_mask[:,:,0] = get_array_mask(edge_mask[:,:,0], sparsity_prob=0.5)

    edge_mask[:,:,1:] = 1 # parameters are considered known.
    if noise: 
        edge_input_features_noisy = copy.deepcopy(edge_input_features)
        node_input_features_noisy = copy.deepcopy(node_input_features)
        
        node_input_features_noisy[:,:,0] = np.random.normal(node_input_features[:,:,0], 0.5/100/4)
        node_input_features_noisy[:,:,1] = np.random.normal(node_input_features[:,:,1], 5/100/4) 

        # add uncertainty to p measurements only not parameters
        edge_input_features_noisy[:,:,0] = np.random.normal(edge_input_features[:,:,0], 5/100/4)

        # convert all arrays to tensors 
        node_input_features_noisy = torch.tensor(node_input_features_noisy, dtype=torch.float32)
        edge_input_features_noisy = torch.tensor(edge_input_features_noisy, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        node_mask = torch.tensor(node_mask, dtype=torch.float32)
        edge_mask = torch.tensor(edge_mask, dtype=torch.float32)

        if tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")
        
        return node_input_features_noisy, node_vapq, edge_input_features_noisy, y_label, y_trafo_label, node_mask, edge_mask, edge_index_dir 

    else: 

        # convert all arrays to tensors 
        node_input_features = torch.tensor(node_input_features, dtype=torch.float32)
        edge_input_features = torch.tensor(edge_input_features, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        node_mask = torch.tensor(node_mask, dtype=torch.float32)
        edge_mask = torch.tensor(edge_mask, dtype=torch.float32)

        if tensor_any_nan(node_input_features, edge_input_features, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")

        return node_input_features, node_vapq, edge_input_features, y_label, y_trafo_label, node_mask, edge_mask, edge_index_dir


#####################################################################################

def load_sc_8_discriminator(net: pp.pandapowerNet, 
                            num_samples: int, 
                            p_true: float,
                            noise: bool = True, 
                            load_p_std: float = 0.3) -> Dict: 
    """
    This dataset is used to test graph-classification of DiffPOOL network in rq_4_gans.ipynb notebook. 
    The goal is to check if the graph-classification works or not.

    Node Features: V, P + Gaussian Noise (if noise) with probability p_true, else Random Node Features
    Edge Features: 1, P^{+} + Gaussian Noise 
    Y label: 1 if node features sampled from V, P else 0 
                
    """
    num_buses = len(net.bus.index)
    num_node_features = 2 # V and P 

    # node input features 
    node_input_features = np.zeros((num_samples, num_buses, num_node_features))

    # edge input features 
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edges = num_lines + num_trafos
    num_edge_features = 1 # p_+ve    

    # edge input features 
    edge_input_features = np.zeros((num_samples, num_edges, num_edge_features))

    # labels 
    y_label = np.zeros((num_samples))

    # variables to permutate in the net 
    pload_ref, qload_ref = copy.deepcopy(net.load['p_mw'].values), copy.deepcopy(net.load['q_mvar'].values)


    # maximum retry: since power flow results cannot converge sometimes 
    max_retries = 20 
    seed_perm = 0

    for iperm in range(num_samples):
        if np.random.rand() <= p_true: 
            retries = 0 
            while retries < max_retries:
                seed_perm += 1 
                try: 
                    rng = np.random.default_rng(seed_perm)
                    # permutate variables 
                    pload = rng.normal(pload_ref, load_p_std)
                    qload = rng.normal(qload_ref, load_p_std)

                    # modify the net data 
                    net.load['p_mw'] = pload 
                    net.load['q_mvar'] = qload 

                    net['converged'] = False 
                    pp.runpp(net, max_iteration=50)

                    # store the results for v and p only 
                    node_input_features[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                    y_label[iperm] = 1
                    node_input_features[iperm,:,1] = np.array(net.res_bus.p_mw.values)

                    # store the results for p+ 
                    edge_input_features[iperm, :, 0] = get_positive_power_flow(net)


                    break # exit while loop if successful (reaching this line.)

                except Exception as e: 
                    print(f"\t Error at permutation {iperm}: {e}")
                    print(f"\t Retry #{retries} at {iperm} with a new random seed...")
                    retries += 1
                    continue

            if retries == max_retries:
                print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
                node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure
        else: 
            node_input_features[iperm,:,0] = np.random.rand((num_buses))
            y_label[iperm] = 0
            node_input_features[iperm,:,1] = np.random.rand((num_buses))
            edge_input_features[iperm,:,0] = np.random.rand((num_edges))
    
    if noise:  
        node_input_features_noisy = copy.deepcopy(node_input_features)
        edge_input_features_noisy = copy.deepcopy(edge_input_features)

        node_input_features_noisy[:,:,0] = np.random.normal(node_input_features[:,:,0], 0.5/100/3)
        node_input_features_noisy[:,:,1] = np.random.normal(node_input_features[:,:,1], 5/100/3) 
        edge_input_features_noisy[:,:,0] = np.random.normal(edge_input_features[:,:,0], 5/100/3)
    
    else: 
        node_input_features_noisy[:,:,0] = node_input_features[:,:,0]
        node_input_features_noisy[:,:,1] = node_input_features[:,:,1]
        edge_input_features_noisy = edge_input_features
    
    node_input_features_noisy = torch.tensor(node_input_features_noisy, dtype=torch.float32)
    edge_input_features_noisy = torch.tensor(edge_input_features_noisy, dtype=torch.float32)
    y_label = torch.tensor(y_label, dtype=torch.float32)

    return node_input_features_noisy, edge_input_features_noisy, y_label 



#####################################################################################

def load_sc_7(net: pp.pandapowerNet, 
              num_samples: int, 
              trafo_ids: List, 
              std_meas: float = 0.1,
              std_pseudo: float = 0.2, 
              load_p_std: float = 0.3) -> Tuple: 
    
    """
    This dataset will consider the pseudo-real measurement infrastructure putting low noise over available measurements and high noise over 
    non-available measurements. 
    """

    assert std_meas <= std_pseudo, "Noise standard deviation over measurements should be less than pseudo-measurements." 

    assert (len(trafo_ids) != 1) | (trafo_ids[0] == "all"), "Total Transformer IDs should be more than 1, otherwise use scenario 1/2/3."

    # check if trafo ids are all MV/LV Trafos 
    # remove switches 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    # node input features  
    num_buses = len(net.bus.index)
    num_node_features = 2 # V and P
    # node input features 
    node_input_features = np.zeros((num_samples, num_buses, num_node_features))

    # edge input features 
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edges = num_lines + num_trafos
    num_edge_features = 9 # p_from, q_from, p_to, q_to, r, x, b, g, tap    

    # edge input features 
    edge_input_features = np.zeros((num_samples, num_edges, num_edge_features))

    # node output label features 
    y_label = np.zeros((num_samples, num_buses, 2)) # v and theta 

    # add r, x, b, g, shift, tap to net 
    net = add_branch_parameters(net)

    # variables to permutate in the net 
    pload_ref, qload_ref = copy.deepcopy(net.load['p_mw'].values), copy.deepcopy(net.load['q_mvar'].values)

    # defective trafo-tap positions 
    # check if the trafo ids exist in net df  
    if trafo_ids != ['all']:
        if len(net.trafo.loc[trafo_ids,:]) == len(trafo_ids):
                if net.trafo.loc[trafo_ids,"name"].str.contains("HV", na=False).any():
                    num_hv_trafos = sum(net.trafo.loc[trafo_ids,"name"].str.contains("HV", na=False)) 
                    nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
                    mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)
                    raise KeyError(f"{num_hv_trafos} Trafos selected are HV. Select out of following indices {mvlv_trafo_ids} \n")
    else: 
        print(f"Selecting all the MV/LV transformers in the network\n")
        nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
        trafo_ids = net.trafo.loc[nonhv_trafo_ids].index


    # mask the trafos so that 
    mask = net.trafo.index.isin(trafo_ids)

    # apply the mask and get the min-max trafo positions 
    tap_min, tap_max = list(net.trafo.loc[mask, "tap_min"]), list(net.trafo.loc[mask, "tap_max"])
    hv_bus_trafos, lv_bus_trafos = list(net.trafo.loc[mask, "hv_bus"]), list(net.trafo.loc[mask, "lv_bus"])

    # trafo tap labels For example: 
    # {sample_id: [{'hv_node': 10,
    #               'lv_node': 5,
    #                'tap_min: 0, # min-max should be non-negative to allow nn.CrossEntropy() to work
    #                'tap_max':4,
    #                'tap_pos':2}, # this will contain tap_pos based on which the pf is solved.
    #  {...}, {...}, ..., {...}]}

    # initialize the trafo label dictionary 
    y_trafo_label = dict()



    
    ### Adding data to the node and edge input feature matrices ###

    # since edge-features for r,x,b,g,shift are not considered to change, assign them here. 
    # Vectorized edge tensor assignment
    
    # lines
    edge_input_features[:,:num_lines,4:8] = np.array(net.line[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32) 
    # trafos
    edge_input_features[:, num_lines:,4:8] = np.array(net.trafo[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32)
    
    # maximum retry: since power flow results cannot converge sometimes 
    max_retries = 20 
    seed_perm = 0 

    for iperm in range(num_samples):

        # set random tap position for edge_input_features tap_position 
        random_tap_for_edge_input = np.random.randint(tap_min, tap_max) 
        net.trafo.loc[mask,"tap_pos"] = random_tap_for_edge_input

        edge_input_features[iperm, num_lines:, 8] = np.array(net.trafo['tap_pos'].values)

        # now set random tap position for power flow calculations 
        random_tap_for_pfr = np.random.randint(tap_min, tap_max)

        net.trafo.loc[mask,"tap_pos"] = random_tap_for_pfr

        # assign labels of trafo 
        y_trafo_label[iperm] = [
            {
                "hv_node": hv,
                "lv_node": lv,
                "tap_min": tmin+tmax,
                "tap_max": tmax+tmax,
                "tap_pos": int(tpos)+tmax
            }
            for hv, lv, tmin, tmax, tpos in zip(hv_bus_trafos, lv_bus_trafos, tap_min, tap_max, random_tap_for_pfr)
        ]

        retries = 0 
        while retries < max_retries:
            seed_perm += 1 
            try: 
                rng = np.random.default_rng(seed_perm)
                # permutate variables 
                pload = rng.normal(pload_ref, load_p_std)
                # qload = rng.normal(qload_ref, load_p_std)

                # modify the net data 
                net.load['p_mw'] = pload 
                # net.load['q_mvar'] = qload 

                net['converged'] = False 
                pp.runpp(net, max_iteration=50)

                # store the results for v and p only 
                node_input_features[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                y_label[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                node_input_features[iperm,:,1] = np.array(net.res_bus.p_mw.values)
                y_label[iperm,:,1] = np.array(net.res_bus.va_degree.values)

                # p_from measurements 
                edge_input_features[iperm,:num_lines,0] = np.array(net.res_line.p_from_mw)
                edge_input_features[iperm,num_lines:,0] = np.array(net.res_trafo.p_hv_mw)

                # q_from measurements 
                edge_input_features[iperm,:num_lines,1] = np.array(net.res_line.q_from_mvar)
                edge_input_features[iperm,num_lines:,1] = np.array(net.res_trafo.q_hv_mvar)

                # p_to measurements 
                edge_input_features[iperm,:num_lines,2] = np.array(net.res_line.p_to_mw)
                edge_input_features[iperm,num_lines:,2] = np.array(net.res_trafo.p_lv_mw)

                # q_to measurements 
                edge_input_features[iperm,:num_lines,3] = np.array(net.res_line.q_to_mvar)
                edge_input_features[iperm,num_lines:,3] = np.array(net.res_trafo.q_lv_mvar)

                break # exit while loop if successful (reaching this line.)

            
            except Exception as e: 
                print(f"\t Error at permutation {iperm}: {e}")
                print(f"\t Retry #{retries} at {iperm} with a new random seed...")
                retries += 1
                continue

        if retries == max_retries:
            print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
            node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure

    # for available measurements add noise of std_meas else add std_pseudo s.t. std_pseudo > std_meas 
    node_input_features_noisy = copy.deepcopy(node_input_features)
    edge_input_features_noisy = copy.deepcopy(edge_input_features)

    # let's perturb all V, P, PQFLOW to std_pseudo 
    node_input_features_noisy = np.random.normal(node_input_features, std_pseudo)
    edge_input_features_noisy[:,:,:4] = np.random.normal(edge_input_features[:,:,:4], std_pseudo)

    # convert y_label to tensor from numpy array
    y_label = torch.tensor(y_label, dtype=torch.float32)

    if net.name == "DFT_TNP": 
        vp_meas_bus = np.array([0,1,2,3,5,7,11,17,22])
        # non_meas_bus_mask = np.ones(num_buses, dtype=bool)
        # non_meas_bus_mask[vp_meas_bus] = False

        pqflow_meas_bus = np.array([5,7,11,17,22])
        
        # apply noise to V, P
        node_input_features_noisy[:, vp_meas_bus, :] = np.random.normal(node_input_features[:, vp_meas_bus, :], std_meas)
        
        # apply noise to PQFLOW: P_in, P_out, Q_in, Q_out only for lines. (because trafo PQFlows are not measured.)
        for iline, line in net.line.iterrows(): 
            from_bus = int(line.from_bus)
            to_bus = int(line.to_bus)

            if to_bus in pqflow_meas_bus:
                edge_input_features_noisy[:, iline, [2,3]] = np.random.normal(edge_input_features_noisy[:, iline, [2,3]], std_meas)

            if from_bus in pqflow_meas_bus: 
                edge_input_features_noisy[:, iline, :2] = np.random.normal(edge_input_features_noisy[:, iline, :2], std_meas)

    else: 
        # consider 50% buses with measurements 
        # randomly sample bus indices out of all buses 
        all_buses = np.array(net.bus.index) 
        fiftyp_bus_mask = np.random.rand(all_buses.size) >= 0.5
        fiftyp_buses = [bus_id for bus_id in all_buses if fiftyp_bus_mask[bus_id]]
        
        # add low std pfr as meas
        node_input_features_noisy[:, fiftyp_buses, :] = np.random.normal(node_input_features[:, fiftyp_buses, :], std_meas)

        # for pqflows 
        for iline, line in net.line.iterrows(): 
            from_bus = int(line.from_bus)
            to_bus = int(line.to_bus)

            if to_bus in fiftyp_buses:
                edge_input_features_noisy[:, iline, [2,3]] = np.random.normal(edge_input_features_noisy[:, iline, [2,3]], std_meas)
            
            if from_bus in fiftyp_buses: 
                edge_input_features_noisy[:, iline, :2] = np.random.normal(edge_input_features_noisy[:, iline, :2], std_meas)

    
    node_input_features_noisy = torch.tensor(node_input_features_noisy, dtype=torch.float32)
    edge_input_features_noisy = torch.tensor(edge_input_features_noisy, dtype=torch.float32)
    
    
    if tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[0]:
        print(f"{tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[1]} has NaNs!")
        raise ValueError("NaN in input data to train!")
    
    return node_input_features_noisy, edge_input_features_noisy, y_label, y_trafo_label

#####################################################################################


def load_sc_6(net: pp.pandapowerNet, 
              num_samples: int, 
              trafo_ids: List,
              noise: bool = True) -> Tuple: 
    """
    This data is considering the real measurement infrastructure for Delft Technopolis only. 
    """
    # if (len(trafo_ids) == 1) & (trafo_ids[0] != "all"): 
    #     raise ValueError("Total Transformer IDs should be more than 1, otherwise use scenario 1/2/3.")

    # check if trafo ids are all MV/LV Trafos 
    # remove switches 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    # node input features  
    num_buses = len(net.bus.index)
    num_node_features = 2 # V and P
    # node input features 
    node_input_features = np.zeros((num_samples, num_buses, num_node_features))
    node_vapq = np.zeros((num_samples, num_buses, 6)) # all V, A, P, Q, P_load, Q_load

    # edge input features 
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edges = num_lines + num_trafos
    num_edge_features = 9 # p_from, q_from, p_to, q_to, r, x, b, g, tap    

    # edge input features 
    edge_input_features = np.zeros((num_samples, num_edges, num_edge_features))

    # node output label features 
    y_label = np.zeros((num_samples, num_buses, 2)) # v and theta 

    # add r, x, b, g, shift, tap to net 
    net = add_branch_parameters(net)

    # variables to permutate in the net 
    pload_ref, qload_ref = copy.deepcopy(net.load['p_mw'].values), copy.deepcopy(net.load['q_mvar'].values)

    # defective trafo-tap positions 
    # check if the trafo ids exist in net df  
    if trafo_ids != ['all']:
        if len(net.trafo.loc[trafo_ids,:]) == len(trafo_ids):
                if net.trafo.loc[trafo_ids,"name"].str.contains("HV", na=False).any():
                    num_hv_trafos = sum(net.trafo.loc[trafo_ids,"name"].str.contains("HV", na=False)) 
                    nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
                    mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)
                    raise KeyError(f"{num_hv_trafos} Trafos selected are HV. Select out of following indices {mvlv_trafo_ids}\n")
    else: 
        print(f"Selecting all the MV/LV transformers in the network \n")
        nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
        trafo_ids = net.trafo.loc[nonhv_trafo_ids].index


    # mask the trafos so that 
    mask = net.trafo.index.isin(trafo_ids)

    # apply the mask and get the min-max trafo positions 
    tap_min, tap_max = list(net.trafo.loc[mask, "tap_min"]), list(net.trafo.loc[mask, "tap_max"])
    hv_bus_trafos, lv_bus_trafos = list(net.trafo.loc[mask, "hv_bus"]), list(net.trafo.loc[mask, "lv_bus"])

    # trafo tap labels For example: 
    # {sample_id: [{'hv_node': 10,
    #               'lv_node': 5,
    #                'tap_min: 0, # min-max should be non-negative to allow nn.CrossEntropy() to work
    #                'tap_max':4,
    #                'tap_pos':2}, # this will contain tap_pos based on which the pf is solved.
    #  {...}, {...}, ..., {...}]}

    # initialize the trafo label dictionary 
    y_trafo_label = dict()
    
    ### Adding data to the node and edge input feature matrices ###

    # since edge-features for r,x,b,g,shift are not considered to change, assign them here. 
    # Vectorized edge tensor assignment
    
    # lines
    edge_input_features[:,:num_lines,4:8] = np.array(net.line[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32) 
    # trafos
    edge_input_features[:, num_lines:,4:8] = np.array(net.trafo[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32)
    
    # maximum retry: since power flow results cannot converge sometimes 
    max_retries = 20 
    seed_perm = 0 

    for iperm in range(num_samples):

        # set random tap position for edge_input_features tap_position 
        random_tap_for_edge_input = np.random.randint(tap_min, tap_max)
        net.trafo.loc[mask,"tap_pos"] = random_tap_for_edge_input.astype("int32")

        edge_input_features[iperm, num_lines:, 8] = np.array(net.trafo['tap_pos'].values)

        # now set random tap position for power flow calculations 
        random_tap_for_pfr = np.random.randint(tap_min, tap_max)

        net.trafo.loc[mask,"tap_pos"] = random_tap_for_pfr.astype("int32")

        # assign labels of trafo 
        y_trafo_label[iperm] = [
            {
                "hv_node": hv,
                "lv_node": lv,
                "tap_min": tmin+tmax,
                "tap_max": tmax+tmax,
                "tap_pos": int(tpos)+tmax
            }
            for hv, lv, tmin, tmax, tpos in zip(hv_bus_trafos, lv_bus_trafos, tap_min, tap_max, random_tap_for_pfr)
        ]

        retries = 0 
        while retries < max_retries:
            seed_perm += 1 
            try: 
                rng = np.random.default_rng(seed_perm)
                # permutate variables 
                pload = rng.normal(pload_ref, net.load.p_std.values)
                # qload = rng.normal(qload_ref, load_p_std)

                # modify the net data 
                net.load['p_mw'] = pload 
                # net.load['q_mvar'] = qload 
                node_vapq[iperm, :, 4][net.load.bus.values] = net.load.p_mw.values
                node_vapq[iperm, :, 5][net.load.bus.values] = net.load.q_mvar.values 

                net['converged'] = False 
                pp.runpp(net, max_iteration=50)

                # store the results for v and p only 
                node_input_features[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                node_vapq[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                y_label[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                node_input_features[iperm,:,1] = np.array(net.res_bus.p_mw.values)
                node_vapq[iperm,:,1] = np.array(net.res_bus.va_degree.values)
                y_label[iperm,:,1] = np.array(net.res_bus.va_degree.values)
                node_vapq[iperm,:,2] = np.array(net.res_bus.p_mw.values)
                node_vapq[iperm,:,3] = np.array(net.res_bus.q_mvar.values)

                # p_from measurements 
                edge_input_features[iperm,:num_lines,0] = np.array(net.res_line.p_from_mw)
                edge_input_features[iperm,num_lines:,0] = np.array(net.res_trafo.p_hv_mw)

                # q_from measurements 
                edge_input_features[iperm,:num_lines,1] = np.array(net.res_line.q_from_mvar)
                edge_input_features[iperm,num_lines:,1] = np.array(net.res_trafo.q_hv_mvar)

                # p_to measurements 
                edge_input_features[iperm,:num_lines,2] = np.array(net.res_line.p_to_mw)
                edge_input_features[iperm,num_lines:,2] = np.array(net.res_trafo.p_lv_mw)

                # q_to measurements 
                edge_input_features[iperm,:num_lines,3] = np.array(net.res_line.q_to_mvar)
                edge_input_features[iperm,num_lines:,3] = np.array(net.res_trafo.q_lv_mvar)

                break # exit while loop if successful (reaching this line.)

            
            except Exception as e: 
                print(f"\t Error at permutation {iperm}: {e}")
                print(f"\t Retry #{retries} at {iperm} with a new random seed...")
                retries += 1
                continue

        if retries == max_retries:
            print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
            node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure

    # calculate the mask for available measurements at node and edge input features 
    # account for real measurements 
    node_mask = np.zeros((num_samples, num_buses, num_node_features))
    edge_mask = np.zeros((num_samples, num_edges, num_edge_features))
    edge_mask[:,:,4:] = 1 # parameters are considered known.
    
    if net.name == 'DFT_TNP':
        node_mask, edge_mask = get_dft_tnp_mask(net, node_mask, edge_mask)
    
    else: 
        sparsity_prob = 0.5
        node_mask = get_array_mask(node_input_features, sparsity_prob=sparsity_prob)
        edge_mask = get_array_mask(edge_input_features, sparsity_prob=sparsity_prob)

    if noise: 
        edge_input_features_noisy = copy.deepcopy(edge_input_features)
        node_input_features_noisy = copy.deepcopy(node_input_features)
        
        node_input_features_noisy[:,:,0] = np.random.normal(node_input_features[:,:,0], 0.5/100/3)
        node_input_features_noisy[:,:,1] = np.random.normal(node_input_features[:,:,1], 5/100/3) 

        # add uncertainty to pq measurements only not parameters
        edge_input_features_noisy[:,:,:4] = np.random.normal(edge_input_features[:,:,:4], 5/100)

        # convert all arrays to tensors 
        node_input_features_noisy = torch.tensor(node_input_features_noisy, dtype=torch.float32)
        edge_input_features_noisy = torch.tensor(edge_input_features_noisy, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        node_mask = torch.tensor(node_mask, dtype=torch.float32)
        edge_mask = torch.tensor(edge_mask, dtype=torch.float32)

        if tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")
        
        return node_input_features_noisy, node_vapq, edge_input_features_noisy, y_label, y_trafo_label, node_mask, edge_mask 

    else: 

        # convert all arrays to tensors 
        node_input_features = torch.tensor(node_input_features, dtype=torch.float32)
        edge_input_features = torch.tensor(edge_input_features, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        node_mask = torch.tensor(node_mask, dtype=torch.float32)
        edge_mask = torch.tensor(edge_mask, dtype=torch.float32)

        if tensor_any_nan(node_input_features, edge_input_features, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features_noisy, edge_input_features_noisy, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")

        return node_input_features, node_vapq, edge_input_features, y_label, y_trafo_label, node_mask, edge_mask

#####################################################################################

def get_dft_tnp_mask(net: pp.pandapowerNet, 
                     node_mask: np.array, 
                     edge_mask: np.array) -> Tuple: 
    """
    Puts 1 where measurements are available for node and edge features. 
    Keeps 1 by default for edge-features that are line/trafo parameters.
    """
    v_at_meas = np.array([0,1,2,3,5,7,11,17,22])
    p_at_meas = v_at_meas

    node_mask[:, v_at_meas, :] = 1 # accounts for both v and p measurements 
    
    # meas at buses 
    pq_flow_meas = np.array([5,7,11,17,22])

    for iline, line in net.line.iterrows(): 
        from_bus = int(line.from_bus)
        to_bus = int(line.to_bus)

        if to_bus in pq_flow_meas: 
            edge_mask[:, iline, [2,3]] = 1 

        if from_bus in pq_flow_meas: 
            edge_mask[:, iline, :2] = 1

    return node_mask, edge_mask


#####################################################################################
#####################################################################################

def get_dft_tnp_mask_sc9(net: pp.pandapowerNet, 
                     node_mask: np.array, 
                     edge_mask: np.array) -> Tuple: 
    """
    Puts 1 where measurements are available for node and edge features. 
    Keeps 1 by default for edge-features that are line/trafo parameters.
    """
    v_at_meas = np.array([0,1,2,3,5,7,11,17,22])
    p_at_meas = v_at_meas

    node_mask[:, v_at_meas, :] = 1 # accounts for both v and p measurements 
    
    # meas at buses 
    pq_flow_meas = np.array([5,7,11,17,22])

    for iline, line in net.line.iterrows():
        from_bus = int(line.from_bus)
        to_bus = int(line.to_bus)

        if to_bus in pq_flow_meas: 
            edge_mask[:, iline, 0] = 1 
        
        if from_bus in pq_flow_meas: 
            edge_mask[:, iline, 0] = 1

    num_lines = len(net.line)

    for itrafo, trafo in net.trafo.iterrows(): 
        hv_bus = int(trafo.hv_bus)
        lv_bus = int(trafo.lv_bus)

        if hv_bus in pq_flow_meas: 
            edge_mask[:, num_lines + itrafo, 0] = 1 
        
        if lv_bus in pq_flow_meas: 
            edge_mask[:, num_lines + itrafo, 0] = 1

    
    return node_mask, edge_mask

#####################################################################################

# Type 1: load label data with permutating loads
def load_sc_1(net:pp.pandapowerNet,  
                num_permutations:int,
                p_std:float,
                trafo_id: List, 
                noise: bool = True) -> Tuple:
    """
    This function 
        - drops switches 
        - adds branch parameters in the network, namely, r, x, b, g, shift, tap from Y-bus matrix (not pandapower equations).
        - gets the min and max tap position for the specified single transformer 
        - randomizes and stores selected trafo tap-pos for power flow 
        - randomizes and stores selected trafo tap-pos as incorrect tap in edge-features input
        - stores node-input features v, theta, p, q as measurements with added gaussian noise to power flow results
        - stores edge-input features r, x, b, g, shift, tap as parameters. 

    """
    assert isinstance(trafo_id, List), "Transformer ID should be a [int]."

    # remove switches 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    # node label tensor 
    num_buses = len(net.bus.index)
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edge_features = 6 # r, x, b, g, shift, tap
    num_node_features = 4 # v, theta, p, q

    # adds r, x, b, g as per-unit values from Y-bus matrix 
    net = add_branch_parameters(net)

    # variables to permute 
    # pgen = net.gen['p_mw'].values
    pload_ref, qload_ref = copy.deepcopy(net.load['p_mw'].values), copy.deepcopy(net.load['q_mvar'].values)

    # input features: [pi, qi, v, theta]
    node_tensor_shape = (num_permutations, num_buses, num_node_features)

    # label: [v, theta]
    y_label = np.zeros([num_permutations, num_buses, 2])

    # select trafos 
    selected_trafo_id = trafo_id

    # ensure if the trafo index exists in the list 
    if len(net.trafo.loc[selected_trafo_id,:]) == len(selected_trafo_id):
        if net.trafo.loc[selected_trafo_id,"name"].str.contains("HV", na=False).any():
            num_hv_trafos = sum(net.trafo.loc[selected_trafo_id,"name"].str.contains("HV", na=False)) 
            nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
            mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)
            raise KeyError(f"{num_hv_trafos} Trafos selected are HV. Select out of following indices {mvlv_trafo_ids}")

    # mask the trafos so that 
    mask = ~net.trafo["name"].str.contains("HV", na = False) & net.trafo.index.isin(selected_trafo_id)

    # apply the mask and get the min-max trafo positions 
    tap_min, tap_max = list(net.trafo.loc[mask, "tap_min"])[0], list(net.trafo.loc[mask, "tap_max"].values)[0]
    hv_trafo, lv_trafo = list(net.trafo.loc[mask, "hv_bus"].values)[0], list(net.trafo.loc[mask, "lv_bus"].values)[0]

    # selected trafo tap_min, tap_max 
    print(f"Selected Trafo {(hv_trafo, lv_trafo)} has Min/Max Tap = {tap_min}/{tap_max} \n")

    # label: trafo tap-pos dict(perm_id: [(hv_trafo, lv_trafo), (tap_min, tap_max) tap_pos]) 
    y_trafo_label = {}

    # edge tensor: [r, x, b, g, tap, shift]
    edge_tensor_shape = (num_permutations, num_lines+num_trafos, num_edge_features)

    node_input_features = np.zeros(node_tensor_shape)
    edge_input_features = np.zeros(edge_tensor_shape)

    # since edge-features for r,x,b,g,shift do not change, assign them here. 
    # Vectorized edge tensor assignment
    # lines
    edge_input_features[:,:num_lines,:4] = np.array(net.line[['r_pu', 'x_pu', 'b_pu', 'g_pu']].values, dtype=np.float32) 
    # trafos
    edge_input_features[:, num_lines:, :5] = np.array(net.trafo[['r_pu', 'x_pu', 'b_pu', 'g_pu', 'shift_rad']].values, dtype=np.float32)
    

    # maximum retry 
    max_retries = 200
    seed_perm = 0

    for iperm in range(num_permutations):

        random_tap_for_edge_input = np.random.randint(tap_min, tap_max)
        net.trafo.loc[mask,"tap_pos"] = random_tap_for_edge_input
        # print(f"\t Trafo connecting buses {(hv_trafo, lv_trafo)} has simulated incorrect tap_pos = {net.trafo.loc[mask,'tap_pos'].values[0]}")

        
        edge_input_features[iperm, num_lines:, 5] = np.array(net.trafo['tap_pos'].values)

        random_tap_for_pfr = np.random.randint(tap_min, tap_max)
        # print(f"\t At iteration {iperm}: Tap-position used for power flow is {tap_min} <= {random_tap_for_pfr} <= {tap_max}")
        
        # assign random tap to net to use in power flow results
        net.trafo.loc[mask, "tap_pos"] = random_tap_for_pfr 

        # assign label trafo as non-negatives 
        y_trafo_label[iperm] = [(hv_trafo, lv_trafo),(tap_min+tap_max, 2*tap_max),random_tap_for_pfr+tap_max]
        # print(f"\t Trafo connecting buses {(hv_trafo, lv_trafo)} has true tap_pos = {random_tap_for_pfr}")

        retries = 0
        while retries < max_retries: 
            seed_perm += 1
            try: 
                rng = np.random.default_rng(seed_perm)
                # permutate variables 
                pload = rng.normal(pload_ref, p_std)
                qload = rng.normal(qload_ref, p_std)    

                # modify the net data 
                net.load['p_mw'] = pload 
                net.load['q_mvar'] = qload 

                net['converged'] = False 
                pp.runpp(net, max_iteration=50)
                # store the results as tensor 
                # node feature tensor: [num_perm, num_nodes, num_features]
                node_input_features[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                y_label[iperm,:,0] = np.array(net.res_bus.vm_pu.values)
                node_input_features[iperm,:,1] = np.array(net.res_bus.va_degree.values)
                y_label[iperm,:,1] = np.array(net.res_bus.va_degree.values)
                node_input_features[iperm,:,2] = np.array(net.res_bus.p_mw.values)
                node_input_features[iperm,:,3] = np.array(net.res_bus.q_mvar.values)
                nominal_sparsity = np.count_nonzero(node_input_features)/node_input_features.size
                break # exit while  if successful

            except Exception as e:
                print(f"\t Error at permutation {iperm}: {e}")
                print(f"\t Retry #{retries} at {iperm} with a new random seed...")
                retries += 1
                continue
            
            
            
        if retries == max_retries: 
            print(f"\t Skipping permutation {iperm} after {max_retries} failed attempts.")
            node_input_features[iperm, :, :] = np.nan  # Assign NaNs to indicate failure

    if noise: 
        # .5 percent noise on voltage, and 5% noise on power and reactive power 
        node_input_features_noisy = copy.deepcopy(node_input_features)
        node_input_features_noisy[:,:,0] = np.random.normal(node_input_features[:,:,0], 0.5/100/3)
        node_input_features_noisy[:,:,2] = np.random.normal(node_input_features[:,:,2], 5/100/3)
        node_input_features_noisy[:,:,3] = np.random.normal(node_input_features[:,:,3], 5/100/3)
    
        

        # convert all arrays to tensors 
        node_input_features_noisy = torch.tensor(node_input_features_noisy, dtype=torch.float32)
        edge_input_features = torch.tensor(edge_input_features, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)

        if tensor_any_nan(node_input_features_noisy, edge_input_features, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features, edge_input_features, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")
        
        return node_input_features_noisy, edge_input_features, y_label, y_trafo_label
    
    else: 
        # convert all arrays to tensors 
        node_input_features = torch.tensor(node_input_features, dtype=torch.float32)
        edge_input_features = torch.tensor(edge_input_features, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)

        if tensor_any_nan(node_input_features, edge_input_features, y_label)[0]:
            print(f"{tensor_any_nan(node_input_features, edge_input_features, y_label)[1]} has NaNs!")
            raise ValueError("NaN in input data to train!")
    
        return node_input_features, edge_input_features, y_label, y_trafo_label
    
#####################################################################################


def load_sc_4(net:pp.pandapowerNet,
                num_permutations:int,
                ) -> Tuple:
    """
    This function is built to test the working of SCNNs in particular.
        - loads a random edge-feature based on number of edges in the net
    """
    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)
    num_edges = num_lines + num_trafos 

    num_edge_features = 6

    num_edge_out_features = 3

    edge_input_feat = torch.rand((num_permutations, num_edges, num_edge_features))

    y_label_edge = torch.zeros((num_permutations, num_edges, num_edge_out_features))

    y_label_edge[:,:,0] = torch.sin(edge_input_feat[:,:,0] + edge_input_feat[:,:,1]**3) # sin(x_0 + x_1^3)
    y_label_edge[:,:,1] = 5 * edge_input_feat[:,:,2] *  edge_input_feat[:,:,3] # x_2 * x_3 * 5
    y_label_edge[:,:,2] = 5 * edge_input_feat[:,:,4] *  edge_input_feat[:,:,5]**3 # x_4 * x_5^3

    print(f"# Num Edges = {edge_input_feat.shape[1]}")

    print(f"# y-label features = {y_label_edge.shape[1]} \n")

    return edge_input_feat, y_label_edge


#####################################################################################


def load_sc_5(net: pp.pandapowerNet, 
              num_permutations:int,
              edge_index: torch.Tensor,
              ) -> Tuple:
    """
    This function is built to check the working of GNN+SCNN regression. Node-data is random, based on which 
    the node-labels and edge-features are calculated. 
    """
    num_nodes = len(net.bus.index)


    node_input_feat = torch.rand((num_permutations, num_nodes, 4))
                
    # Create 2D targets (num_samples, num_nodes, 2) that depend on inputs
    y = torch.zeros((num_permutations, num_nodes, 2))

    # First output dimension: sin of feature sum (original pattern)
    y[:, :, 0] = torch.sin(node_input_feat.sum(dim=-1))

    # Second output dimension: cos of feature product (new pattern)
    y[:, :, 1] = torch.cos(node_input_feat[:, :, 0] * node_input_feat[:, :, 1] / node_input_feat[:, :, 2]) * node_input_feat[:,:,3]

    
    edge_attr = create_patterned_edge_attributes(node_input_feat, edge_index=edge_index)
    return node_input_feat, edge_attr, y

# Edge attributes
# Create synthetic edge attributes with patterns from node data
def create_patterned_edge_attributes(x: torch.Tensor, edge_index: torch.Tensor):
    num_samples, _, _ = x.shape
    num_edges = edge_index.shape[1]
    
    num_edge_features = 2

    # Initialize edge attributes
    edge_attr = torch.zeros((num_samples, num_edges, num_edge_features))
    
    for i in range(num_samples):
        # Pattern 1: Difference between connected nodes' features
        src, dst = edge_index
        feature_diff = x[i, src, 0] - x[i, dst, 1]  # Using different features for src/dst
        edge_attr[i, :, 0] = torch.sin(feature_diff)  # Non-linear transform
        
        # Pattern 2: Product of connected nodes' features
        feature_product = x[i, src, 2] * x[i, dst, 3]
        edge_attr[i, :, 1] = torch.cos(feature_product)  # Different non-linear transform
        
        # Add some random noise (optional)
        edge_attr[i] += 0.1 * torch.randn_like(edge_attr[i])
    
    return edge_attr


#####################################################################################


def retrieve_trafo_minmaxedge(sample_data: Data) -> Tuple: 

    # for a single sample 
    y_trafo_label_1 = sample_data.y_trafo_label

    tap_min = [trafo["tap_min"] for trafo in y_trafo_label_1]
    tap_max = [trafo["tap_max"] for trafo in y_trafo_label_1]
    trafo_edge = [(trafo["hv_node"], trafo["lv_node"]) for trafo in y_trafo_label_1]

    return np.array(tap_min), np.array(tap_max), np.array(trafo_edge)
    
def trafo_batch_collate(trafo_hop: int, # TODO 
                        loader: DataLoader):
    """Using the y_trafo_label in the batch and trafo_hop used in the model, 
     this function calculates the trafo_hop neighbors to the trafo_edge and 
     stores it."""
    for batch in loader:
        batch[0]._store.abc = 4
        node_data = batch[0] 
        
        # get the terminal buses of the specific trafo
        trafo_edge = node_data.y_trafo_label[0][0]

        # since batch can have multiple graphs 
        num_graphs = int(len(batch[0].ptr)-1)

        for i in range(num_graphs):
            # for toy network e.g., ptr = tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
            # get node indices for the current graph
            start, end = node_data.ptr[i], node_data.ptr[i+1] # 0, 9 as first graph, based on which trafo_id gets calculated 
            
            # get edges for the current graph 
            mask = (node_data.edge_index[0] >= start) & (node_data.edge_index[0] < end)

            # current graph in batch 
            curr_G_edges = node_data.edge_index[:, mask] - start 

            # build graph 
            curr_G = nx.Graph()
            curr_G_elist = curr_G_edges.T.tolist()
            curr_G.add_edges_from(curr_G_elist)

            # group nodes by shortest path length 
            hop_group_dict_all = {h: [] for h in range(trafo_hop + 1)}
            for node in trafo_edge: # terminal nodes 
                shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=trafo_hop)
                
                for neighbor, hop in shortest_hops.items():
                    if hop > 0 and neighbor not in trafo_edge:
                        hop_group_dict_all[hop].append(int(neighbor + start))
                    else:
                        hop_group_dict_all[hop].append(int(neighbor + start))
            
            # only consider new and unique nodes for next hop starting with terminal nodes for 0 hop 
            hop_group_dict_uniq = {key: [] for key in hop_group_dict_all.keys()}
            all_uniq_nodes = set() # unique nodes considered for all hops 

            for key in hop_group_dict_all.keys(): 
                hop_uniq_nodes = set(hop_group_dict_all[key]) # no repeatition 
                # if uniq nodes at hop are not already in all_uniq nodes
                if not all_uniq_nodes & hop_uniq_nodes: 
                    all_uniq_nodes.update(hop_group_dict_all[key])
                    hop_group_dict_uniq[key] = hop_group_dict_all[key]
                else: 
                    for and_node in list(all_uniq_nodes & hop_uniq_nodes): 
                        hop_uniq_nodes.remove(and_node)
                    all_uniq_nodes.update(hop_uniq_nodes)
                    hop_group_dict_uniq[key] = list(hop_uniq_nodes)
        
        batch[0].trafo_hop_uniq_neighbors = hop_group_dict_uniq
        print(batch[0])
    return loader 



