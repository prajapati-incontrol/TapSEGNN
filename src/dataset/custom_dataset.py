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
                 sampled_input_data: Dict, 
                 missing_measurements: bool = False): 
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

        # when giving input to generator model 
        self.missing_measurements = missing_measurements
        self.node_mask = sampled_input_data['node_mask']
        self.edge_mask = sampled_input_data['edge_mask']

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

        # generator input 
        if missing_measurements: 
            # random noise over missing values 
            self.z_x = torch.rand_like(self.x)
            self.z_edge_attr = torch.rand_like(self.edge_attr)

            # available measurements with random noise at missing values 
            self.x_bar = self.x * self.node_mask + (1 - self.node_mask) * self.z_x 
            self.edge_attr_bar = self.edge_attr * self.edge_mask + (1 - self.edge_mask) * self.z_edge_attr

    def __getitem__(self, index):
        if not self.missing_measurements: 
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
        
        else: 
            node_graph_data = Data(x=self.x_bar[index],
                                    edge_index=self.edge_index, 
                                    y=self.y[index], 
                                    y_trafo_label=self.y_trafo_label[index],
                                    y_tap = self.y_tap[index,:],
                                    node_mask=self.node_mask[index], 
                                    x_pfr=self.x[index])
            if self.dir_pf: 
                edge_HL_graph_data = Data(x=self.edge_attr_bar[index],
                                        edge_index=self.edge_index_l[index], 
                                        edge_attr=self.edge_weight_l[index],
                                        edge_index_u=self.edge_index_u[index], 
                                        edge_attr2=self.edge_weight_u[index], 
                                        edge_mask=self.edge_mask[index], 
                                        x_pfr=self.edge_attr[index])
            else:
                edge_HL_graph_data = Data(x=self.edge_attr_bar[index],
                                        edge_index=self.edge_index_l, 
                                        edge_attr=self.edge_weight_l,
                                        edge_index_u=self.edge_index_u, 
                                        edge_attr2=self.edge_weight_u,
                                        edge_mask=self.edge_mask[index], 
                                        x_pfr=self.edge_attr[index])

            edge_LG_graph_data = Data(x=self.edge_attr_bar[index], 
                                    edge_index=self.edge_index_lg, 
                                    edge_attr=self.edge_weight_lg, 
                                    edge_mask=self.edge_mask[index], 
                                    x_pfr=self.edge_attr[index])

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

