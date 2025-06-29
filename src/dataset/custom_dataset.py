from torch_geometric.data import Data, Dataset 
from torch.utils.data import Dataset as torch_dataset
import numpy as np
import torch 
from typing import Literal, Dict
import os 
import sys
import time 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.gen_utils import get_edge_index_lu, get_edge_index_lg

#####################################################################################

class DiscDataset(Dataset): 
    def __init__(self, 
                 sampled_input_data: Dict):
        """
        Dataset for just discriminator training. 
        
        """
        super().__init__()
        self.x = sampled_input_data['node_input_feat']
        self.x1 = sampled_input_data['edge_input_feat']
        self.y = sampled_input_data['y_label']
        self.edge_index = sampled_input_data['edge_index']
        self.num_samples = self.x.shape[0]

    def __getitem__(self, idx):
        return Data(x=self.x[idx], 
                    edge_index=self.edge_index,
                    edge_attr=self.x1[idx],
                    y=self.y[idx],
                    )
    
    def __len__(self):
        return self.num_samples

#####################################################################################

class NodeEdgeTapDatasetV2(Dataset):
    def __init__(self,
                 model_name: str, 
                 sampled_input_data: Dict): 
        """
        Make torch_geometric dataset for node, edge and tap input features and labels.

        node_labels: Tensor of shape (num_samples, num_nodes, num_node_features)
        edge_labels: Tensor of shape (num_samples, num_edges, num_edge_features)
        y_label: Tensor of shape (num_samples, num_nodes, 2)
        edge_index: Tensor of shape (2, num_edges)
        y_trafo_label: Dictionary containing {iperm: (hv_bus, lv_bus), tap_pos} for all permutations
        """
        super().__init__()
        print(f"Dataset for {model_name} selected!\n")
        self.x = sampled_input_data['node_input_feat'] 
        self.y = sampled_input_data['y_label'] 
        self.num_samples = self.x.shape[0] # num_samples
        self.edge_index = sampled_input_data['edge_index'] # same for all samples 
        if sampled_input_data['edge_index_dir']:
            self.dir_pf = True
            print("\n Directed power flows accounted in dataset...\n")
            self.edge_index_dir = sampled_input_data['edge_index_dir']
            edge_index_lu_outputs = get_edge_index_lu(self.edge_index_dir)
        else: 
            self.dir_pf = False
            print("\n Directed power flows NOT accounted in dataset...\n")
            edge_index_lu_outputs = get_edge_index_lu(self.edge_index)
        time.sleep(3)
        self.y_trafo_label = sampled_input_data['y_trafo_label'] 

        # hodge-laplacian dataset
        self.edge_attr = sampled_input_data['edge_input_feat'] 
        
        self.edge_index_l = edge_index_lu_outputs[0]
        self.edge_index_u = edge_index_lu_outputs[1]
        self.edge_weight_l = edge_index_lu_outputs[2]
        self.edge_weight_u = edge_index_lu_outputs[3]
        

        # linegraph laplacian dataset
        self.edge_index_lg = get_edge_index_lg(self.edge_index)[0] # line graph laplacian
        self.edge_weight_lg = get_edge_index_lg(self.edge_index)[1] 



        self.num_trafos = len(self.y_trafo_label[0]) # any sample

        self.y_tap = torch.zeros((self.num_samples, 1, self.num_trafos), dtype=torch.long)

        for sample in list(self.y_trafo_label.keys()):
            for trafo in range(self.num_trafos):
                self.y_tap[sample, :, trafo] = self.y_trafo_label[sample][trafo]['tap_pos'] 


    def __getitem__(self, index):
        node_graph_data = Data(x=self.x[index],
                                edge_index=self.edge_index, 
                                y=self.y[index], 
                                y_trafo_label=self.y_trafo_label[index],
                                y_tap = self.y_tap[index,:])
        if self.dir_pf: 
            edge_HL_graph_data = Data(x=self.edge_attr[index],
                                    edge_index=self.edge_index_l[index], 
                                    edge_attr=self.edge_weight_l[index],
                                    edge_index_u=self.edge_index_u[index], 
                                    edge_attr2=self.edge_weight_u[index])
        else:
            edge_HL_graph_data = Data(x=self.edge_attr[index],
                                    edge_index=self.edge_index_l, 
                                    edge_attr=self.edge_weight_l,
                                    edge_index_u=self.edge_index_u, 
                                    edge_attr2=self.edge_weight_u) 

        edge_LG_graph_data = Data(x=self.edge_attr[index], 
                                edge_index=self.edge_index_lg, 
                                edge_attr=self.edge_weight_lg)

        return node_graph_data, edge_HL_graph_data, edge_LG_graph_data

    def __len__(self):
        return self.num_samples
    
#####################################################################################
class GenDataset(NodeEdgeTapDatasetV2): 
    def __init__(self, 
                 model_name: str, 
                 sampled_input_data: Dict): 
        """
        Dataset for GANs training.
        """
        super().__init__(model_name = model_name,
                         sampled_input_data=sampled_input_data)
        self.node_mask = sampled_input_data['node_mask']
        self.edge_mask = sampled_input_data['edge_mask']
        self.y_gan = torch.zeros((self.num_samples,), dtype=torch.float32)
        self.y_fool = torch.ones((self.num_samples,), dtype=torch.float32)

        # # random noise over missing values 
        self.z_x = torch.rand_like(self.x)
        self.z_edge_attr = torch.rand_like(self.edge_attr)

        # # available measurements with random noise at missing values 
        self.x_bar = self.x * self.node_mask + (1 - self.node_mask) * self.z_x 
        self.edge_attr_bar = self.edge_attr * self.edge_mask + (1 - self.edge_mask) * self.z_edge_attr


    def __getitem__(self, index):
        node_graph_data = Data(x=self.x_bar[index],
                                edge_index=self.edge_index, 
                                y=self.y_gan[index],
                                y_fool=self.y_fool[index], 
                                node_mask=self.node_mask[index],
                                x_pfr=self.x[index])
        edge_HL_graph_data = Data(x=self.edge_attr_bar[index],
                                edge_index=self.edge_index_l[index], 
                                edge_attr=self.edge_weight_l[index],
                                edge_mask=self.edge_mask[index],
                                edge_index_u=self.edge_index_u[index], 
                                edge_attr2=self.edge_weight_u[index], 
                                x_pfr=self.edge_attr[index])
        edge_LG_graph_data = Data(x=self.edge_attr_bar[index], 
                                edge_index=self.edge_index_lg, 
                                edge_attr=self.edge_weight_lg,
                                edge_mask=self.edge_mask[index],
                                x_pfr=self.edge_attr[index])
        return node_graph_data, edge_HL_graph_data, edge_LG_graph_data
    
    def __len__(self): 
        return super().__len__()








#####################################################################################

class NodeEdgeTapDataset(Dataset):
    def __init__(self, 
                 model_name: str, 
                 sampled_input_data: Dict): 
        """
        Make torch_geometric dataset for node, edge and tap input features and labels.

        node_labels: Tensor of shape (num_samples, num_nodes, num_node_features)
        edge_labels: Tensor of shape (num_samples, num_edges, num_edge_features)
        y_label: Tensor of shape (num_samples, num_nodes, 2)
        edge_index: Tensor of shape (2, num_edges)
        y_trafo_label: Dictionary containing {iperm: (hv_bus, lv_bus), tap_pos} for all permutations
        """
        super().__init__() 
        self.model_name = model_name
        self.multitapse_models = {"MultiTapSEGNN","NEGATRegressor"}
        self.tapse_models = {"TapSEGNN","TapGNN","TapNRegressor","NERegressor", "NRegressor", "TAGNRegressor"}
        self.se_models = {"NERegressor", "NRegressor", "TAGNRegressor","NEGATRegressor"}
        self.edger_models = {"EdgeRegressor", "EdgeLGRegressor"}
        if model_name in self.tapse_models: # also add model below 
            print(f"Dataset for {model_name} selected!\n")
            self.x = sampled_input_data['node_input_feat'] 
            self.y = sampled_input_data['y_label'] 
            self.num_samples = self.x.shape[0] # num_samples
            self.edge_index = sampled_input_data['edge_index'] # same for all samples 

           
            # hodge-laplacian dataset
            self.edge_attr = sampled_input_data['edge_input_feat'] 
            self.edge_index_l = get_edge_index_lu(self.edge_index)[0]
            self.edge_index_u = get_edge_index_lu(self.edge_index)[1]
            self.edge_weight_l = get_edge_index_lu(self.edge_index)[2]
            self.edge_weight_u = get_edge_index_lu(self.edge_index)[3]
            

            # linegraph laplacian dataset
            self.edge_index_lg = get_edge_index_lg(self.edge_index)[0] # line graph laplacian
            self.edge_weight_lg = get_edge_index_lg(self.edge_index)[1] 

            if model_name not in self.se_models: 
                self.is_trafo_bool = sampled_input_data['y_trafo_label']            
                if self.is_trafo_bool:
                    self.y_trafo_label = sampled_input_data['y_trafo_label']
                    self.y_tap = torch.tensor([label[2] for label in list(self.y_trafo_label.values())], dtype=torch.long)
            

        elif model_name in self.edger_models: 
            print(f"Dataset for {model_name} selected!\n")
            self.edge_attr = sampled_input_data['edge_input_feat']
            self.y = sampled_input_data['y_label_edge'] 
            self.edge_index = sampled_input_data['edge_index']
            self.num_samples = sampled_input_data['edge_input_feat'].shape[0]
            
            # hodge-laplacian dataset
            self.edge_index_l = get_edge_index_lu(self.edge_index)[0]
            self.edge_index_u = get_edge_index_lu(self.edge_index)[1]
            self.edge_weight_l = get_edge_index_lu(self.edge_index)[2]
            self.edge_weight_u = get_edge_index_lu(self.edge_index)[3]
            
            # linegraph laplacian dataset
            self.edge_index_lg = get_edge_index_lg(self.edge_index)[0] # line graph laplacian
            self.edge_weight_lg = get_edge_index_lg(self.edge_index)[1] 

            # edge labels for case 4,5 
            self.y_label_edge = sampled_input_data['y_label_edge']

        elif model_name in self.multitapse_models: 
            print(f"Dataset for {model_name} selected!\n")
            self.x = sampled_input_data['node_input_feat'] 
            self.y = sampled_input_data['y_label']
            self.num_samples = self.x.shape[0] # num_samples
            self.edge_index = sampled_input_data['edge_index'] # same for all samples
            self.edge_index_dir = sampled_input_data['edge_index_dir'] # may not be same for all samples (depends on power flow directions from power flow results) 
           
            # hodge-laplacian dataset
            self.edge_attr = sampled_input_data['edge_input_feat'] 
            self.edge_index_l = get_edge_index_lu(self.edge_index_dir)[0]
            self.edge_index_u = get_edge_index_lu(self.edge_index_dir)[1]
            self.edge_weight_l = get_edge_index_lu(self.edge_index_dir)[2]
            self.edge_weight_u = get_edge_index_lu(self.edge_index_dir)[3]
            

            # linegraph laplacian dataset
            self.edge_index_lg = get_edge_index_lg(self.edge_index)[0] # line graph laplacian
            self.edge_weight_lg = get_edge_index_lg(self.edge_index)[1] 

            self.y_trafo_label = sampled_input_data['y_trafo_label']        

            if self.y_trafo_label:
                self.num_trafos = len(self.y_trafo_label[0]) # any sample

                self.y_tap = torch.zeros((self.num_samples, 1, self.num_trafos), dtype=torch.long)

                for sample in list(self.y_trafo_label.keys()):
                    for trafo in range(self.num_trafos):
                        self.y_tap[sample, :, trafo] = self.y_trafo_label[sample][trafo]['tap_pos'] 
        else: 
            raise NameError("Invalid model name! Pretraining left to add.")

    def __getitem__(self, index):
        if self.model_name in self.tapse_models:
            # num_edges = self.edge_attr[index].shape[0]

            # # trafo_mask 
            # trafo_mask = torch.arange(num_edges) >= self.num_lines
            if self.model_name not in self.se_models: 
                node_graph_data = Data(x=self.x[index],
                                    edge_index=self.edge_index, 
                                    y=self.y[index], 
                                    y_trafo_label=self.y_trafo_label[index],
                                    y_tap = self.y_tap[index])
            else: 
                node_graph_data = Data(x=self.x[index],
                                    edge_index=self.edge_index, 
                                    y=self.y[index])
            edge_HL_graph_data = Data(x=self.edge_attr[index],
                                      edge_index=self.edge_index_l, 
                                      edge_attr=self.edge_weight_l,
                                      edge_index_u=self.edge_index_u, 
                                      edge_attr2=self.edge_weight_u)
            edge_LG_graph_data = Data(x=self.edge_attr[index], 
                                      edge_index=self.edge_index_lg, 
                                      edge_attr=self.edge_weight_lg)

            return node_graph_data, edge_HL_graph_data, edge_LG_graph_data
        
        elif self.model_name in self.edger_models: 
            edge_HL_graph_data = Data(x=self.edge_attr[index],
                                      edge_index=self.edge_index_l, 
                                      edge_attr=self.edge_weight_l,
                                      edge_index_u=self.edge_index_u, 
                                      edge_attr2=self.edge_weight_u,
                                      y=self.y_label_edge[index])
            edge_LG_graph_data = Data(x=self.edge_attr[index], 
                                      edge_index=self.edge_index_lg, 
                                      edge_attr=self.edge_weight_lg,
                                      y=self.y_label_edge[index])
            
            return edge_HL_graph_data, edge_LG_graph_data
        

        elif self.model_name in self.multitapse_models: 
            if self.y_trafo_label: 
                node_graph_data = Data(x=self.x[index],
                                    edge_index=self.edge_index, 
                                    y=self.y[index], 
                                    y_trafo_label=self.y_trafo_label[index],
                                    y_tap = self.y_tap[index,:])
            else: 
                node_graph_data = Data(x=self.x[index],
                                    edge_index=self.edge_index, 
                                    y=self.y[index])
            edge_HL_graph_data = Data(x=self.edge_attr[index],
                                      edge_index=self.edge_index_l[index], 
                                      edge_attr=self.edge_weight_l[index],
                                      edge_index_u=self.edge_index_u[index], 
                                      edge_attr2=self.edge_weight_u[index])
            edge_LG_graph_data = Data(x=self.edge_attr[index], 
                                      edge_index=self.edge_index_lg, 
                                      edge_attr=self.edge_weight_lg)

            return node_graph_data, edge_HL_graph_data, edge_LG_graph_data

    def __len__(self):
        return self.num_samples

#####################################################################################

class FCNNDataset(torch_dataset):
    def __init__(self,  
                 sampled_input_data: Dict):
        """ Make pytorch dataset for node features and labels without taking topology into account."""
        super().__init__()
        self.X_mat = sampled_input_data['node_input_feat']
        self.Y_mat = sampled_input_data['y_label']

    def __len__(self):
        return len(self.X_mat)

    def __getitem__(self, index):
        self.X = self.X_mat[index].flatten()
        self.Y = self.Y_mat[index].flatten()
        return self.X, self.Y     

# class NodeEdgeTapDataset(Dataset):
#     def __init__(self, 
#                  model_name: str, 
#                  sampled_input_data: Dict): 
#         """
#         Make torch_geometric dataset for node, edge and tap input features and labels.

#         node_labels: Tensor of shape (num_samples, num_nodes, num_node_features)
#         edge_labels: Tensor of shape (num_samples, num_edges, num_edge_features)
#         y_label: Tensor of shape (num_samples, num_nodes, 2)
#         edge_index_list: Tensor of shape (2, num_edges)
#         num_lines: Integer row index from which edges are transformers in edge_labels or last row representing edge as line
#         y_trafo_label: Dictionary containing {iperm: (hv_bus, lv_bus), tap_pos} for all permutations
#         """
#         super().__init__() 
#         self.model_name = model_name
#         if model_name in {"TapGNN","TapNRegressor","NERegressor", "NRegressor", "TAGNRegressor4SE"}:
#             print(f"Datast for {model_name} selected!\n")
#             self.x = sampled_input_data['node_input_feat'] 
#             self.y = sampled_input_data['y_label'] # TODO: se labels rather than power flow labels.
#             self.num_samples = self.x.shape[0] # num_samples
#             self.edge_index = sampled_input_data['edge_index_list'] # same for all samples 
#             self.edge_attr = sampled_input_data['edge_input_feat']
#             self.edge_index_l, self.edge_index_u = get_edge_index_lu(self.edge_index)

#             self.edge_index_lg = get_edge_index_lg(self.edge_index) # line graph laplacian 
#             self.num_lines = sampled_input_data['num_lines'] 
#             self.y_trafo_label = sampled_input_data['y_trafo_label']
#             self.y_tap = torch.tensor([label[2] for label in list(self.y_trafo_label.values())], dtype=torch.long)
        
        
#         elif model_name in {"EdgeRegressor", "EdgeLGRegressor"}: 
#             self.y = sampled_input_data['y_label_edge'] # TODO: se labels rather than power flow labels.
#             self.edge_index = sampled_input_data['edge_index_list']
#             self.edge_attr = sampled_input_data['edge_input_feat']
#             self.num_samples = sampled_input_data['edge_input_feat'].shape[0]
#             self.edge_index_l, self.edge_index_u = get_edge_index_lu(self.edge_index)
#             self.edge_index_lg = get_edge_index_lg(self.edge_index) # line graph laplacian 
#             self.num_nodes = self.edge_attr.shape[1]
        
        
#         else: 
#             raise NameError("Invalid model name! Pretraining left to add.")

#     def __getitem__(self, index):
#         if self.model_name in {"TapGNN","TapNRegressor","NERegressor", "NRegressor", "TAGNRegressor4SE"}:
#             # num_edges = self.edge_attr[index].shape[0]

#             # # trafo_mask 
#             # trafo_mask = torch.arange(num_edges) >= self.num_lines

#             return Data(x=self.x[index], 
#                         edge_index = self.edge_index, 
#                         edge_attr=self.edge_attr[index], 
#                         y=self.y[index], 
#                         edge_index_l=self.edge_index_l,
#                         edge_index_u=self.edge_index_u,
#                         edge_index_lg=self.edge_index_lg,
#                         y_trafo_label=self.y_trafo_label[index],
#                         # trafo_mask = trafo_mask, 
#                         # num_lines = self.num_lines, 
#                         y_tap = self.y_tap[index])
#         elif self.model_name in {"EdgeRegressor", "EdgeLGRegressor"}: 
#             return Data(x=self.edge_attr[index], 
#                         edge_index=self.edge_index, # nodes edge index used in model to create hodge laplacian
#                         y=self.y[index],
#                         edge_index_l=self.edge_index_l,
#                         edge_index_u=self.edge_index_u,
#                         edge_index_lg=self.edge_index_lg,
#                         num_nodes=self.num_nodes)
        

#     def __len__(self):
#         return self.num_samples


