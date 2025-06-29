import torch.nn as nn 
from torch.nn import LayerNorm
from torch_geometric.nn.conv import TAGConv, MessagePassing, GATConv
from torch_geometric.nn.dense import DenseGCNConv
from typing import Tuple, Literal
import torch
import os, sys
import torch.nn.functional as F
from typing import List 
import numpy as np
import networkx as nx 
from math import ceil 
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.dense import dense_diff_pool

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.gen_utils import tensor_any_nan

#########################################################################################################
#########################################################################################################
   
class NEGATRegressor(nn.Module):
    """ Only SE with Node and Edge Regression followed by GATConv."""
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 list_edge_hidden_features: list, 
                 gat_out_features: int, 
                 gat_head: int,
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','cpu','mps'] = 'cpu',
    ):
        super().__init__()
        self.name = "NEGATRegressor" # used in logging 
        self.bias = bias
        self.device = device

        ###### GNN: node regression convolution layers ###### 
        self.node_layers = nn.ModuleList()
        in_feats_n = node_input_features 
         
        if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
            for idx, hid_feats_n in enumerate(list_node_hidden_features): 
                self.node_layers.append(TAGConv(in_channels=in_feats_n, 
                                                out_channels=hid_feats_n, 
                                                K=k_hop_node, 
                                                bias=bias, 
                                                normalize=adj_norm))
                # no normalization after last layer
                if normalize and idx < len(list_node_hidden_features): 
                    self.node_layers.append(LayerNorm(hid_feats_n))
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fc_node = nn.Linear(hid_feats_n, node_out_features)

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = nn.ModuleList()
        self.edge_biases = nn.ParameterList()

        in_feats_e = edge_input_features

        # add bias to SCNN as a whole (rather than individual TAGConv above)
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features):

                self.edge_layers.append(nn.ModuleList([TAGConv(in_channels=in_feats_e, 
                                                out_channels=hid_feats_e,
                                                K=k_hop_edge,
                                                bias=False,
                                                normalize=False),
                                        TAGConv(in_channels=in_feats_e,
                                                out_channels=hid_feats_e, 
                                                K=k_hop_edge,
                                                bias=False, # TODO: True? 
                                                normalize=False)]))
                if bias:
                    self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
                else: 
                    self.edge_biases.append(None)
                in_feats_e = hid_feats_e
        else: 
            hid_feats_e = in_feats_e

        self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

        self.gatconv = GATConv(in_channels=node_out_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_output_features)

        # self.gatconv = GATConv(in_channels=node_input_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_input_features)

        # since gatconv with multiple heats concatenates outputs, a final regression layer is required. 
        # mlp 
        self.mlp_gat = nn.Sequential(
            nn.Linear(gat_out_features * self.gatconv.heads, 2, bias = True),
            # nn.ReLU(), # TODO: Dropout required?
        )

        self.reset_parameters() 
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        # nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> torch.Tensor:
        node_data, edge_data = tupleData[0], tupleData[1]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_data.x.to(self.device)
        edge_index_l = edge_data.edge_index.to(self.device)
        edge_weight_l = edge_data.edge_attr.to(self.device)
        edge_index_u = edge_data.edge_index_u.to(self.device)
        edge_weight_u = edge_data.edge_attr2.to(self.device)

        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                x = F.relu(x)
                # torch.tanh_(x)
            else: 
                x = layer(x)

        x = self.fc_node(x)

        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            x1 = layer[0](x=x1, edge_index=edge_index_l, edge_weight=edge_weight_l) \
                #   + layer[1](x=x1, edge_index=edge_index_u, edge_weight=edge_weight_u)
            
            if self.bias: 
                x1 += e_bias
            torch.relu_(x1) 

        x1 = self.fc_edge(x1)
        
        # gatconv
        alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

        # agg_mssgs to SE predictions
        x_o = self.mlp_gat(alpha_gat)

        # return x_o
        return x_o

#########################################################################################################
#########################################################################################################

class GATRegressor(nn.Module):
    """GATConv followed by a Linear layer."""
    def __init__(self, 
                 gat_in_features: int, 
                 gat_edge_features: int, 
                 gat_out_features: int, 
                 gat_head: int, 
                 bias: bool = True,  
                 device: Literal['cuda','cpu','mps'] = 'cpu',
                 ):
        
        super().__init__()
        self.name = "GATRegressor"
        self.bias = bias 
        self.device = device 

        self.gatconv = GATConv(in_channels=gat_in_features,
                               out_channels=gat_out_features, 
                               heads=gat_head, 
                               edge_dim=gat_edge_features, 
                               )
        
        self.mlp_gat = nn.Sequential(
            nn.Linear(gat_out_features * self.gatconv.heads, 2, bias = True),
            # nn.ReLU(),
        )

    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        # nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)


    def forward(self, tupleData) -> Tuple[torch.Tensor]: 
        node_data, edge_data = tupleData[0], tupleData[1]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_data.x.to(self.device)


        # gatconv
        alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

        # agg_mssgs to SE predictions
        x_o = self.mlp_gat(alpha_gat)

        return x_o


#########################################################################################################
#########################################################################################################


class NGATRegressor(nn.Module):
    """TAGConv + GATConv followed by a Linear layer."""
    def __init__(self,
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int,
                 k_hop_node: int, 
                 gat_edge_features: int, 
                 gat_out_features: int, 
                 gat_head: int, 
                 bias: bool = True,
                 normalize = True,  
                 device: Literal['cuda','cpu','mps'] = 'cpu',
                 ):
        
        super().__init__()
        self.name = "NGATRegressor"
        self.bias = bias 
        self.device = device 

        ###### GNN: node regression convolution layers ###### 
        self.node_layers = nn.ModuleList()
        in_feats_n = node_input_features 
         
        if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
            for idx, hid_feats_n in enumerate(list_node_hidden_features): 
                self.node_layers.append(TAGConv(in_channels=in_feats_n, 
                                                out_channels=hid_feats_n, 
                                                K=k_hop_node, 
                                                bias=bias, 
                                                normalize=True))
                # no normalization after last layer
                if normalize and idx < len(list_node_hidden_features): 
                    self.node_layers.append(LayerNorm(hid_feats_n))
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fc_node = nn.Linear(hid_feats_n, node_out_features)

        self.gatconv = GATConv(in_channels=node_out_features,
                               out_channels=gat_out_features, 
                               heads=gat_head, 
                               edge_dim=gat_edge_features, 
                               )
        
        self.mlp_gat = nn.Sequential(
            nn.Linear(gat_out_features * self.gatconv.heads, 2, bias = True),
            # nn.ReLU(),
        )

    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        # nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)


    def forward(self, tupleData) -> Tuple[torch.Tensor]: 
        node_data, edge_data = tupleData[0], tupleData[1]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_data.x.to(self.device)

        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                x = F.relu(x)
                # torch.tanh_(x)
            else: 
                x = layer(x)

        x = self.fc_node(x)
        
        # gatconv
        alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

        # agg_mssgs to SE predictions
        x_o = self.mlp_gat(alpha_gat)

        return x_o

#########################################################################################################
#########################################################################################################

class NEGATGenerator(NEGATRegressor):
    """
    Generator model based on NEGATRegressor. 
    
    """
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 list_edge_hidden_features: list, 
                 gat_out_features: int, 
                 gat_head: int,
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','cpu','mps'] = 'cpu',
    ):
        super().__init__(node_input_features=node_input_features,
                         list_node_hidden_features=list_node_hidden_features,
                         node_out_features=node_out_features,
                         k_hop_node=k_hop_node,
                         edge_input_features=edge_input_features,
                         edge_output_features=edge_output_features,
                         k_hop_edge=k_hop_edge,
                         list_edge_hidden_features=list_edge_hidden_features,
                         gat_out_features=gat_out_features,
                         gat_head=gat_head,
                         bias=bias,
                         normalize=normalize,
                         adj_norm=adj_norm,
                         device=device
                         )
        self.name = "NEGATGenerator"
        self.bias = bias 
        self.device = device

        super().reset_parameters()
    
    def forward(self, tupleData, mask: bool = True) -> Tuple[torch.Tensor]:
        node_data, edge_data = tupleData[0], tupleData[1]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_data.x.to(self.device)
        edge_index_l = edge_data.edge_index.to(self.device)
        edge_weight_l = edge_data.edge_attr.to(self.device)

        node_mask = node_data.node_mask.to(self.device)
        edge_mask = edge_data.edge_mask.to(self.device)

        
        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index).relu()
            else: 
                x = layer(x)

        x = self.fc_node(x)

        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            x1 = layer[0](x=x1, edge_index=edge_index_l, edge_weight=edge_weight_l) \
                #   + layer[1](x=x1, edge_index=edge_index_u, edge_weight=edge_weight_u)
            
            if self.bias: 
                x1 += e_bias
            torch.relu_(x1) 

        # output imputed values 
        x1 = self.fc_edge(x1).flatten()

        # gatconv
        alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

        # agg_mssgs to imputations 
        x_o = self.mlp_gat(alpha_gat)

        if mask: 
            x_o = x_o * node_mask + x_o * (1 - node_mask)
            x1_o = x1 * edge_mask[:,0] + x1 * (1 - edge_mask[:,0])
        else: 
            x_o = x_o 
            x1_o = x1

        return x_o, x1_o
    
#########################################################################################################
#########################################################################################################

class NEGATGenerator_LGL(NEGATRegressor): 
    """
    """
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 list_edge_hidden_features: list, 
                 gat_out_features: int, 
                 gat_head: int,
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','cpu','mps'] = 'cpu',
    ):
        super().__init__(node_input_features=node_input_features,
                         list_node_hidden_features=list_node_hidden_features,
                         node_out_features=node_out_features,
                         k_hop_node=k_hop_node,
                         edge_input_features=edge_input_features,
                         edge_output_features=edge_output_features,
                         k_hop_edge=k_hop_edge,
                         list_edge_hidden_features=list_edge_hidden_features,
                         gat_out_features=gat_out_features,
                         gat_head=gat_head,
                         bias=bias,
                         normalize=normalize,
                         adj_norm=adj_norm,
                         device=device
                         )
        self.name = "NEGATRegressor_LGL"
        self.bias = bias 
        self.device = device 

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = nn.ModuleList()
        self.edge_biases = nn.ParameterList()

        in_feats_e = edge_input_features

        # add bias to SCNN as a whole (rather than individual TAGConv above)
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features):

                self.edge_layers.append(TAGConv(in_channels=in_feats_e, 
                                                out_channels=hid_feats_e,
                                                K=k_hop_edge,
                                                bias=False,
                                                normalize=False))
                if bias:
                    self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
                else: 
                    self.edge_biases.append(None)
                in_feats_e = hid_feats_e
        else: 
            hid_feats_e = in_feats_e

        self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

        super().reset_parameters() 

    def forward(self, tupleData, mask: bool = True) -> Tuple[torch.Tensor, torch.Tensor]: 
        node_data, edge_lg_data = tupleData[0], tupleData[2]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_lg_data.x.to(self.device)
        # linegraph laplacian edge indices
        edge_index_lg = edge_lg_data.edge_index.to(self.device)

        node_mask = node_data.node_mask.to(self.device)
        edge_mask = edge_lg_data.edge_mask.to(self.device)

        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                x = F.relu(x)
                # torch.tanh_(x)
            else: 
                x = layer(x)

        x = self.fc_node(x)
        
        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            x1 = layer(x1, edge_index_lg) # this is where linegraph laplacian gets implemented. 
            if self.bias: 
                x1 += e_bias
            torch.relu_(x1) 

        x1 = self.fc_edge(x1).flatten()
        
        # gatconv
        alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

        # agg_mssgs to SE predictions
        x_o = self.mlp_gat(alpha_gat)

        if mask: 
            x_o = x_o * node_mask + x_o * (1 - node_mask)
            x1_o = x1 * edge_mask[:,0] + x1 * (1 - edge_mask[:,0])
        else: 
            x_o = x_o 
            x1_o = x1

        return x_o, x1_o

#########################################################################################################
#########################################################################################################


class DiffPoolDiscriminator(MessagePassing):
    def __init__(self, 
                 in_channel: int, 
                 hidden_channel: int, 
                 out_channel: int, 
                 num_nodes: int, 
                 pooling_ratio: float = 0.5):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel 

        # embded layers 
        self.gnn_embed1 = DenseGCNConv(in_channel, hidden_channel)
        self.gnn_embed2 = DenseGCNConv(hidden_channel, hidden_channel)
        self.gnn_embed3 = DenseGCNConv(hidden_channel, out_channel)

        # pool layers 
        pool_1_clusters = ceil(pooling_ratio * num_nodes)
        self.gnn_pool1 = DenseGCNConv(in_channel, pool_1_clusters)

        pool_2_clusters = ceil(pooling_ratio * pool_1_clusters)
        self.gnn_pool2 = DenseGCNConv(hidden_channel, pool_2_clusters)

        # final pool all ones
        self.pool_final = torch.ones((pool_2_clusters, 1))

        # binary classification 
        self.linear = nn.Linear(out_channel, 1)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.gnn_embed1.reset_parameters()
        self.gnn_embed2.reset_parameters()
        self.gnn_embed3.reset_parameters()

        self.gnn_pool1.reset_parameters()
        self.gnn_pool2.reset_parameters()
        self.linear.reset_parameters()


    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:  
        x, _ = to_dense_batch(x, batch)
        
        adj = to_dense_adj(edge_index=edge_index,
                           edge_attr=edge_attr.flatten(), 
                           batch=batch)
        
        z = self.gnn_embed1(x, adj).relu()
        s = self.gnn_pool1(x, adj)
        
        x2, adj2, _, _ = dense_diff_pool(x=z, 
                                       adj=adj, 
                                       s=s)
        
        z2 = self.gnn_embed2(x2, adj2).relu()
        s2 = self.gnn_pool2(x2, adj2)

        x3, adj3, _, _ = dense_diff_pool(x=z2, 
                                       adj=adj2, 
                                       s=s2)
        
        z3 = self.gnn_embed3(x3, adj3).relu()
        s_final = self.pool_final

        x4, _, _, _ = dense_diff_pool(x=z3, 
                                     adj=adj3, 
                                     s=s_final)
        
        x4 = x4.squeeze(1) # [batch_size, out_channels/logit size]

        out = self.linear(x4).squeeze(1)

        return out

#########################################################################################################
#########################################################################################################

class PredTapModel(nn.Module): 
    """
    Predicting Tap-position using SE outputs as inputs. 
    """
    def __init__(self, 
                 trafo_hop: int, 
                 num_trafo_neighbors: List, 
                 trafo_out_features: List, 
                 bias: bool = True):
        super().__init__()
        self.trafo_hop = trafo_hop 
        if all(feat == trafo_out_features[0] for feat in trafo_out_features):
            print(f"All transformers have same number of taps, using shared parameter \omega.")

            # consider all transformer have same number of taps  
            self.multi_trafo_mlp_o = nn.ModuleList(
                [ nn.ModuleList(
                    [
                        nn.Linear(2, 16, bias=bias), 
                        nn.Linear(16, int(trafo_out_features[0]), bias=bias)
                    ]
                ) for _ in range(trafo_hop + 1)
                ]
            )
        else: 
            raise NotImplementedError("Not implemented for transformer with different number of tap classes.")
    
        # self.trafo_mlp_a = nn.Linear(trafo_out_features, 1, bias = False) # output logits 

        # since every transformer can have different number of neighbors 
        self.multi_trafo_mlp_a = nn.ModuleDict() 
        for trafo_id, num_trafo_neighbors_for_trafo_id in enumerate(num_trafo_neighbors):
            self.multi_trafo_mlp_a["trafo a - "+str(trafo_id)] = nn.Linear(int(num_trafo_neighbors_for_trafo_id), 1, bias=True)
        
        self.reset_parameters()

    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                # nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData, trained_se_model):
        node_data = tupleData[0]

        x_o = trained_se_model(tupleData)

        multi_trafo_pred = dict()

        # since the trafo_edges (hv_node, lv_node) will be the same for all the transformers across all the samples 
        # in y_trafo_label select first sample to enumerate trafo_id and trafo_edges

        for trafo_id, trafo_dict in enumerate(node_data.y_trafo_label[0]): 
            
            # get terminal buses for trafo_id trafo 
            trafo_edge = (trafo_dict["hv_node"], trafo_dict["lv_node"])
            
            # since batch can have multiple graphs 
            batch_size = node_data.ptr.size(0) -1 
            
            single_trafo_preds = [] # batch_size, logits 

            for i in range(batch_size):
                # tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
                # get node indices for the current graphg 
                start, end = node_data.ptr[i], node_data.ptr[i+1]
                curr_G_x = x_o[start:end] 
                
                # get edges for the current graph 
                mask = (node_data.edge_index[0] >= start) & (node_data.edge_index[0] < end)
                curr_G_edges = node_data.edge_index[:, mask] - start # since start will be nonzero for ptr > 0

                # build graph 
                curr_G = nx.Graph()
                curr_G_elist = curr_G_edges.T.tolist()
                curr_G.add_edges_from(curr_G_elist)

                # group nodes by shortest path length 
                hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
                for node in trafo_edge:  # Terminal nodes (trafo_edge)
                    shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                    
                    for neighbor, hop in shortest_hops.items():
                        if hop > 0 and neighbor not in trafo_edge:
                            hop_group_dict_all[hop].append(neighbor)
                        else:
                            hop_group_dict_all[hop].append(neighbor)
                
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
                
                # agg and concat 
                hop_stack = []
                for h in range(self.trafo_hop + 1):
                    if len(hop_group_dict_uniq[h]) > 0:
                        neighbor_feats = curr_G_x[hop_group_dict_uniq[h]]  # Select features for nodes at hop h
                        omega_x = self.multi_trafo_mlp_o[h][0](neighbor_feats)  # Shared parameters 
                        omega_x = self.multi_trafo_mlp_o[h][1](omega_x)
                        hop_stack.append(omega_x)

                # Concatenate and transform
                hop_concat = torch.cat(hop_stack, dim=0).t()  # Shape: (feature_dim, num_nodes)
                trafo_pred = self.multi_trafo_mlp_a["trafo a - "+str(trafo_id)](hop_concat).t()  # Shape: (num_nodes, output_dim)
                
                single_trafo_preds.append(trafo_pred)
            
            # combine all keeping batch-wise structure 
            all_G_trafo_preds = torch.cat(single_trafo_preds, dim=0)
        
            multi_trafo_pred[trafo_id] = all_G_trafo_preds

        return multi_trafo_pred
    
    
#########################################################################################################
#########################################################################################################



class MultiTapSEGNN(NEGATRegressor):
    """
    Predicting multi/all tap-positions and performing SE using GATConv after GNN+SCNN.
    
    This model is an extension of TapSEGNN. 
    """
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int, 
                 edge_input_features: int, 
                 list_edge_hidden_features: List[int],
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 trafo_hop: int, 
                 num_trafo_neighbors: List, 
                 trafo_out_features: List, # number of tap classes
                 gat_out_features: int,  
                 gat_head: int,  
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # adjacency not normalized for SCNN (returns NaNs)
                 device: Literal['cuda','mps','cpu'] = 'cpu',
                 ): 
        super().__init__(node_input_features=node_input_features,
                         list_node_hidden_features=list_node_hidden_features,
                         node_out_features=node_out_features,
                         k_hop_node=k_hop_node,
                         edge_input_features=edge_input_features,
                         edge_output_features=edge_output_features,
                         k_hop_edge=k_hop_edge,
                         list_edge_hidden_features=list_edge_hidden_features,
                         gat_out_features=gat_out_features,
                         gat_head=gat_head,
                         bias=bias,
                         normalize=normalize,
                         adj_norm=adj_norm,
                         device=device
                         )
        
        self.name = "MultiTapSEGNN"
        self.bias = bias 
        self.trafo_hop = trafo_hop
        self.device = device

        # TODO: Maybe use different parameters for each transformer???
        # So that, trafo_mlp_o and trafo_mlp_a will be different for each transformer...

        # note: trafo out features are the number of classes 
        if all(feat == trafo_out_features[0] for feat in trafo_out_features):
            print(f"All transformers have same number of taps, using shared parameter \omega.")

            # consider all transformer have same number of taps  
            # self.multi_trafo_mlp_o = nn.ModuleList(
            #     [
            #         nn.Linear(2, int(trafo_out_features[0]), bias = bias) for _ in range(trafo_hop + 1)
            #     ]
            # )
            self.multi_trafo_mlp_o = nn.ModuleList(
                [ nn.ModuleList(
                    [
                        nn.Linear(2, 16, bias=bias), 
                        nn.Linear(16, int(trafo_out_features[0]), bias=bias)
                    ]
                ) for _ in range(trafo_hop + 1)
                ]
            )
        else: 
            raise NotImplementedError("Not implemented for transformer with different number of tap classes.")
    
        # self.trafo_mlp_a = nn.Linear(trafo_out_features, 1, bias = False) # output logits 

        # since every transformer can have different number of neighbors 
        self.multi_trafo_mlp_a = nn.ModuleDict() 
        for trafo_id, num_trafo_neighbors_for_trafo_id in enumerate(num_trafo_neighbors):
            self.multi_trafo_mlp_a["trafo a - "+str(trafo_id)] = nn.Linear(int(num_trafo_neighbors_for_trafo_id), 1, bias=bias)
        
        super().reset_parameters()



    def forward(self, tupleData):
        node_data = tupleData[0]

        x_o = super().forward(tupleData)

        multi_trafo_pred = dict()

        # since the trafo_edges (hv_node, lv_node) will be the same for all the transformers across all the samples 
        # in y_trafo_label select first sample to enumerate trafo_id and trafo_edges

        for trafo_id, trafo_dict in enumerate(node_data.y_trafo_label[0]): 
            
            # get terminal buses for trafo_id trafo 
            trafo_edge = (trafo_dict["hv_node"], trafo_dict["lv_node"])
            
            # since batch can have multiple graphs 
            batch_size = node_data.ptr.size(0) -1 
            
            single_trafo_preds = [] # batch_size, logits 

            for i in range(batch_size):
                # tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
                # get node indices for the current graphg 
                start, end = node_data.ptr[i], node_data.ptr[i+1]
                curr_G_x = x_o[start:end] 
                
                # get edges for the current graph 
                mask = (node_data.edge_index[0] >= start) & (node_data.edge_index[0] < end)
                curr_G_edges = node_data.edge_index[:, mask] - start # since start will be nonzero for ptr > 0

                # build graph 
                curr_G = nx.Graph()
                curr_G_elist = curr_G_edges.T.tolist()
                curr_G.add_edges_from(curr_G_elist)

                # group nodes by shortest path length 
                hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
                for node in trafo_edge:  # Terminal nodes (trafo_edge)
                    shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                    
                    for neighbor, hop in shortest_hops.items():
                        if hop > 0 and neighbor not in trafo_edge:
                            hop_group_dict_all[hop].append(neighbor)
                        else:
                            hop_group_dict_all[hop].append(neighbor)
                
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
                
                # agg and concat 
                hop_stack = []
                for h in range(self.trafo_hop + 1):
                    if len(hop_group_dict_uniq[h]) > 0:
                        neighbor_feats = curr_G_x[hop_group_dict_uniq[h]]  # Select features for nodes at hop h
                        omega_x = self.multi_trafo_mlp_o[h][0](neighbor_feats)  # Shared parameters 
                        omega_x = self.multi_trafo_mlp_o[h][1](omega_x)
                        hop_stack.append(omega_x)

                # Concatenate and transform
                hop_concat = torch.cat(hop_stack, dim=0).t()  # Shape: (feature_dim, num_nodes)
                trafo_pred = self.multi_trafo_mlp_a["trafo a - "+str(trafo_id)](hop_concat).t()  # Shape: (num_nodes, output_dim)
                
                single_trafo_preds.append(trafo_pred)
            
            # combine all keeping batch-wise structure 
            all_G_trafo_preds = torch.cat(single_trafo_preds, dim=0)
        
            multi_trafo_pred[trafo_id] = all_G_trafo_preds

        return x_o, multi_trafo_pred

#########################################################################################################
#########################################################################################################




class TapSEGNN(NEGATRegressor):
    def __init__(self,
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 list_edge_hidden_features: List[int],
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 trafo_hop: int,
                 num_trafo_neighbors: int, 
                 trafo_out_features: int, # number of tap classes
                 gat_out_features: int, 
                 gat_head: int,
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','mps','cpu'] = 'cpu',
    ):
        super().__init__(node_input_features=node_input_features,
                         list_node_hidden_features=list_node_hidden_features,
                         node_out_features=node_out_features,
                         k_hop_node=k_hop_node,
                         edge_input_features=edge_input_features,
                         edge_output_features=edge_output_features,
                         k_hop_edge=k_hop_edge,
                         list_edge_hidden_features=list_edge_hidden_features,
                         gat_out_features=gat_out_features,
                         gat_head=gat_head,
                         bias=bias,
                         normalize=normalize,
                         adj_norm=adj_norm,
                         device=device)
        self.name = "TapSEGNN" # used in logging 
        self.trafo_hop = trafo_hop 
        self.bias = bias 
        self.device = device
        
        # TODO: Maybe use different parameters for each transformer???
        # So that, trafo_mlp_o and trafo_mlp_a will be different for each transformer...

        self.trafo_mlp_o = nn.ModuleList(
            [
                nn.Linear(2, trafo_out_features, bias=False) for _ in range(trafo_hop + 1)
            ]
        )

        # self.trafo_mlp_a = nn.Linear(trafo_out_features, 1, bias = False) # output logits 
        self.trafo_mlp_a = nn.Linear(num_trafo_neighbors, 1, bias = False) # output logits 

        super().reset_parameters() 
    
    def forward(self, tupleData):
        node_data = tupleData[0]

        x_o = super().forward(tupleData)

        # get the terminal buses of the specific trafo  
        trafo_edge = node_data.y_trafo_label[0][0]

        # since batch can have multiple graphs 
        batch_size = node_data.ptr.size(0)-1
        trafo_preds = [] # batch_size, logits 

        for i in range(batch_size):
            # tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
            # get node indices for the current graphg 
            start, end = node_data.ptr[i], node_data.ptr[i+1]
            curr_G_x = x_o[start:end] 
            
            # get edges for the current graph 
            mask = (node_data.edge_index[0] >= start) & (node_data.edge_index[0] < end)
            curr_G_edges = node_data.edge_index[:, mask] - start # since start will be nonzero for ptr > 0

            # build graph 
            curr_G = nx.Graph()
            curr_G_elist = curr_G_edges.T.tolist()
            curr_G.add_edges_from(curr_G_elist)

            # group nodes by shortest path length 
            hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
            for node in trafo_edge:  # Terminal nodes (trafo_edge)
                shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                
                for neighbor, hop in shortest_hops.items():
                    if hop > 0 and neighbor not in trafo_edge:
                        hop_group_dict_all[hop].append(neighbor)
                    else:
                        hop_group_dict_all[hop].append(neighbor)
            
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
            
            # agg and concat 
            hop_stack = []
            for h in range(self.trafo_hop + 1):
                if len(hop_group_dict_uniq[h]) > 0:
                    neighbor_feats = curr_G_x[hop_group_dict_uniq[h]]  # Select features for nodes at hop h
                    omega_x = self.trafo_mlp_o[h](neighbor_feats)  # Shared MLP
                    hop_stack.append(omega_x)

            # Concatenate and transform
            hop_concat = torch.cat(hop_stack, dim=0).t()  # Shape: (feature_dim, num_nodes)
            trafo_pred = self.trafo_mlp_a(hop_concat).t()  # Shape: (num_nodes, output_dim)
            
            trafo_preds.append(trafo_pred)
        
        # combine all keeping batch-wise structure 
        all_G_trafo_preds = torch.cat(trafo_preds, dim=0)
        

        return x_o, all_G_trafo_preds






#########################################################################################################

class TapGNN(nn.Module):
    """Tap and SE but with agg concat operator."""
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 list_edge_hidden_features: list,
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 trafo_hop: int,
                 num_trafo_neighbors: int, 
                 trafo_out_features: int,
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','mps','cpu'] = 'cpu',
    ):
        super().__init__()
        self.name = "TapGNN" # used in logging 
        self.bias = bias
        self.trafo_hop = trafo_hop
        self.device = device

        ###### GNN: node regression convolution layers ###### 
        self.node_layers = nn.ModuleList()
        in_feats_n = node_input_features 
         
        if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
            for idx, hid_feats_n in enumerate(list_node_hidden_features): 
                self.node_layers.append(TAGConv(in_channels=in_feats_n, 
                                                out_channels=hid_feats_n, 
                                                K=k_hop_node, 
                                                bias=bias, 
                                                normalize=adj_norm))
                # no normalization after last layer
                if normalize and idx < len(list_node_hidden_features): 
                    self.node_layers.append(LayerNorm(hid_feats_n))
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fc_node = nn.Linear(hid_feats_n, node_out_features)

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = nn.ModuleList()
        self.edge_biases = nn.ParameterList()

        in_feats_e = edge_input_features

        # add bias to SCNN as a whole (rather than individual TAGConv above)
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features):

                self.edge_layers.append(nn.ModuleList([TAGConv(in_channels=in_feats_e, 
                                                out_channels=hid_feats_e,
                                                K=k_hop_edge,
                                                bias=False,
                                                normalize=False),
                                        TAGConv(in_channels=in_feats_e,
                                                out_channels=hid_feats_e, 
                                                K=k_hop_edge,
                                                bias=False, # TODO: True? 
                                                normalize=False)]))
                if bias:
                    self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
                in_feats_e = hid_feats_e

        self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

        self.mlp_agg = nn.Sequential(
            nn.Linear(2*node_out_features + edge_output_features, 2, bias = True),
            nn.ReLU(), # TODO: Dropout required?
        )

        self.trafo_mlp_o = nn.ModuleList(
            [
                nn.Linear(2, trafo_out_features, bias=False) for _ in range(trafo_hop + 1)
            ]
        )

        # self.trafo_mlp_a = nn.Linear(trafo_out_features, 1, bias = False) # output logits 
        self.trafo_mlp_a = nn.Linear(num_trafo_neighbors, 1, bias = False) # output logits 

        self.reset_parameters() 
    
    def sum_concat(self, 
                   x: torch.Tensor, 
                   edge_index: torch.Tensor, 
                   edge_attr: torch.Tensor) -> torch.Tensor: 
        # [num_nodes, 2*node_out_features + edge_out_features]
        x, edge_index, edge_attr = x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device)
        agg_mssgs = torch.zeros((x.shape[0], 2*x.shape[1] + edge_attr.shape[1]))
        
        for e_id, (u, v) in enumerate(edge_index.T): 
            y_hat_i, y_hat_j = x[u], x[v]
            edge_ij = edge_attr[e_id]

            agg_mssgs[u] += torch.cat([y_hat_i, y_hat_j, edge_ij], dim=-1)
            agg_mssgs[v] += torch.cat([y_hat_j, y_hat_i, edge_ij], dim=-1)
        
        return agg_mssgs
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        # nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
        node_data, edge_data = tupleData[0], tupleData[1]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_data.x.to(self.device)
        edge_index_l = edge_data.edge_index.to(self.device)
        edge_weight_l = edge_data.edge_attr.to(self.device)
        edge_index_u = edge_data.edge_index_u.to(self.device)
        edge_weight_u = edge_data.edge_attr2.to(self.device)

        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                # x = F.relu(x)
                torch.tanh_(x)
            else: 
                x = layer(x)
        x = self.fc_node(x)
        

        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            x1 = layer[0](x1, edge_index_l, edge_weight_l) + layer[1](x1, edge_index_u, edge_weight_u)
            if self.bias: 
                x1 += e_bias
            torch.relu_(x1) 

        x1 = self.fc_edge(x1)
        
        # aggregate messages for state-estimation X^o 
        agg_mssgs = self.sum_concat(x, edge_index, x1)

        # agg_mssgs to SE predictions
        x_o = self.mlp_agg(agg_mssgs) 

        # get the terminal buses of the specific trafo  
        trafo_edge = node_data.y_trafo_label[0][0]

        # since batch can have multiple graphs 
        batch_size = node_data.ptr.size(0)-1
        trafo_preds = [] # batch_size, logits 

        for i in range(batch_size):
            # tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
            # get node indices for the current graphg 
            start, end = node_data.ptr[i], node_data.ptr[i+1]
            curr_G_x = x_o[start:end] 
            

            # get edges for the current graph 
            mask = (node_data.edge_index[0] >= start) & (node_data.edge_index[0] < end)
            curr_G_edges = node_data.edge_index[:, mask] - start # since start will be nonzero for ptr > 0

            # build graph 
            curr_G = nx.Graph()
            curr_G_elist = curr_G_edges.T.tolist()
            curr_G.add_edges_from(curr_G_elist)

            # group nodes by shortest path length 
            hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
            for node in trafo_edge:  # Terminal nodes (trafo_edge)
                shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                
                for neighbor, hop in shortest_hops.items():
                    if hop > 0 and neighbor not in trafo_edge:
                        hop_group_dict_all[hop].append(neighbor)
                    else:
                        hop_group_dict_all[hop].append(neighbor)
            
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
            

            
            # agg and concat 
            hop_stack = []
            for h in range(self.trafo_hop + 1):
                if len(hop_group_dict_uniq[h]) > 0:
                    neighbor_feats = curr_G_x[hop_group_dict_uniq[h]]  # Select features for nodes at hop h
                    omega_x = self.trafo_mlp_o[h](neighbor_feats)  # Shared MLP
                    hop_stack.append(omega_x)

            # Concatenate and transform
            hop_concat = torch.cat(hop_stack, dim=0).t()  
            # print(f"Shape of hop_concat = {hop_concat.shape}")
            trafo_pred = self.trafo_mlp_a(hop_concat).t() 
            # print(f"Shape of trafo_pred = {trafo_pred.shape}")
            
            trafo_preds.append(trafo_pred)
        
        # combine all keeping batch-wise structure 
        all_G_trafo_preds = torch.cat(trafo_preds, dim=0)
        
        return x_o, all_G_trafo_preds





# #########################################################################################################




class NERegressor(nn.Module):
    """ Only SE with Node and Edge Regression."""
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 list_edge_hidden_features: list, 
                 agg_op: str = "sum",
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','cpu','mps'] = 'cpu',
    ):
        super().__init__()
        self.name = "NERegressor" # used in logging 
        self.bias = bias
        self.device = device
        self.agg_op = agg_op

        ###### GNN: node regression convolution layers ###### 
        self.node_layers = nn.ModuleList()
        in_feats_n = node_input_features 
         
        if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
            for idx, hid_feats_n in enumerate(list_node_hidden_features): 
                self.node_layers.append(TAGConv(in_channels=in_feats_n, 
                                                out_channels=hid_feats_n, 
                                                K=k_hop_node, 
                                                bias=bias, 
                                                normalize=adj_norm))
                # no normalization after last layer
                if normalize and idx < len(list_node_hidden_features): 
                    self.node_layers.append(LayerNorm(hid_feats_n))
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fc_node = nn.Linear(hid_feats_n, node_out_features)

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = nn.ModuleList()
        self.edge_biases = nn.ParameterList()

        in_feats_e = edge_input_features

        # add bias to SCNN as a whole (rather than individual TAGConv above)
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features):

                self.edge_layers.append(nn.ModuleList([TAGConv(in_channels=in_feats_e, 
                                                out_channels=hid_feats_e,
                                                K=k_hop_edge,
                                                bias=False,
                                                normalize=False),
                                        TAGConv(in_channels=in_feats_e,
                                                out_channels=hid_feats_e, 
                                                K=k_hop_edge,
                                                bias=False, # TODO: True? 
                                                normalize=False)]))
                if bias:
                    self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
                else: 
                    self.edge_biases.append(None)
                in_feats_e = hid_feats_e
        else: 
            hid_feats_e = in_feats_e

        self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

        # mlp 
        self.mlp_agg = nn.Sequential(
            nn.Linear(2*node_out_features + edge_output_features, 2, bias = True),
            nn.ReLU(), # TODO: Dropout required?
        )

        self.reset_parameters() 
    
    def agg_concat(self, 
                   x: torch.Tensor, 
                   edge_index: torch.Tensor, 
                   edge_attr: torch.Tensor, 
                   agg_op: str = "sum") -> torch.Tensor: 
        # [num_nodes, 2*node_out_features + edge_out_features]
        x, edge_index, edge_attr = x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device)


        # agg mssgs of shape: # [num_nodes, 2*node_feats + edge_feats]
        agg_mssgs = torch.zeros((x.shape[0], 2*x.shape[1] + edge_attr.shape[1])).to(device=self.device)
        
        
        for e_id, (u, v) in enumerate(edge_index.T): 
            y_hat_i, y_hat_j = x[u], x[v]
            edge_ij = edge_attr[e_id]

            agg_mssgs[u] += torch.cat([y_hat_i, y_hat_j, edge_ij], dim=-1)
            agg_mssgs[v] += torch.cat([y_hat_j, y_hat_i, edge_ij], dim=-1)
            
        if agg_op == "mean":
            G = nx.Graph()
            G.add_edges_from(list((int(u), int(v)) for (u, v) in edge_index.T))
            for node, deg in G.degree(): 
                agg_mssgs[node] = agg_mssgs[node] / torch.tensor(deg, dtype=torch.float32)


        return agg_mssgs
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        # nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
        node_data, edge_data = tupleData[0], tupleData[1]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_data.x.to(self.device)
        edge_index_l = edge_data.edge_index.to(self.device)
        edge_weight_l = edge_data.edge_attr.to(self.device)
        edge_index_u = edge_data.edge_index_u.to(self.device)
        edge_weight_u = edge_data.edge_attr2.to(self.device)

        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                # x = F.relu(x)
                torch.tanh_(x)
            else: 
                x = layer(x)
        x = self.fc_node(x)

        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            x1 = layer[0](x=x1, edge_index=edge_index_l, edge_weight=edge_weight_l) \
                  + layer[1](x=x1, edge_index=edge_index_u, edge_weight=edge_weight_u)
            
            if self.bias: 
                x1 += e_bias
            torch.relu_(x1) 

        x1 = self.fc_edge(x1)
        
        # aggregate messages for state-estimation X^o 
        agg_mssgs = self.agg_concat(x, edge_index, x1, agg_op=self.agg_op)

        # agg_mssgs to SE predictions
        x_o = self.mlp_agg(agg_mssgs) 

        return x_o 

# #########################################################################################################

class NRegressor(nn.Module): 
    """Only GNN Regressor for SE but with agg_concat excluding Edge Features."""
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','cpu','mps'] = 'cpu',
    ):
        super().__init__()
        self.name = "NRegressor" # used in logging 
        self.bias = bias
        self.device = device

        ###### GNN: node regression convolution layers ###### 
        self.node_layers = nn.ModuleList()
        in_feats_n = node_input_features 
         
        if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
            for idx, hid_feats_n in enumerate(list_node_hidden_features): 
                self.node_layers.append(TAGConv(in_channels=in_feats_n, 
                                                out_channels=hid_feats_n, 
                                                K=k_hop_node, 
                                                bias=bias, 
                                                normalize=adj_norm))
                # no normalization after last layer
                if normalize and idx < len(list_node_hidden_features): 
                    self.node_layers.append(LayerNorm(hid_feats_n))
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fc_node = nn.Linear(hid_feats_n, node_out_features)

        
        # mlp 
        self.mlp_agg = nn.Sequential(
            nn.Linear(2*node_out_features, 2, bias = True),
            nn.ReLU(), # TODO: Dropout required?
        )

        self.reset_parameters() 
    
    def sum_concat_n(self, 
                   x: torch.Tensor, 
                   edge_index: torch.Tensor) -> torch.Tensor: 
        # [num_nodes, 2*node_out_features + edge_out_features]
        x, edge_index = x.to(self.device), edge_index.to(self.device)

        agg_mssgs = torch.zeros((x.shape[0], 2*x.shape[1])).to(device=self.device)
        
        for e_id, (u, v) in enumerate(edge_index.T): 
            y_hat_i, y_hat_j = x[u], x[v]

            agg_mssgs[u] += torch.cat([y_hat_i, y_hat_j], dim=-1)
            agg_mssgs[v] += torch.cat([y_hat_j, y_hat_i], dim=-1)
        
        return agg_mssgs
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        # nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
        node_data = tupleData[0]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)


        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                # x = F.relu(x)
                torch.tanh_(x)
            else: 
                x = layer(x)
        x = self.fc_node(x)
        
        # aggregate messages for state-estimation X^o 
        agg_mssgs = self.sum_concat_n(x, edge_index)

        # agg_mssgs to SE predictions
        x_o = self.mlp_agg(agg_mssgs) 

        return x_o 




# #########################################################################################################

class TapNRegressor(nn.Module): 
    """Node Regression with GNN and Tap prediction. No edge regression."""
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 trafo_hop: int,
                 num_trafo_neighbors: int, 
                 trafo_out_features: int,
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','mps','cpu'] = 'cpu',
    ):
        super().__init__()
        self.name = "TapNRegressor" # used in logging 
        self.bias = bias
        self.trafo_hop = trafo_hop
        self.device = device

        ###### GNN: node regression convolution layers ###### 
        self.node_layers = nn.ModuleList()
        in_feats_n = node_input_features 
         
        if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
            for idx, hid_feats_n in enumerate(list_node_hidden_features): 
                self.node_layers.append(TAGConv(in_channels=in_feats_n, 
                                                out_channels=hid_feats_n, 
                                                K=k_hop_node, 
                                                bias=bias, 
                                                normalize=adj_norm))
                # no normalization after last layer
                if normalize and idx < len(list_node_hidden_features): 
                    self.node_layers.append(LayerNorm(hid_feats_n))
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fc_node = nn.Linear(hid_feats_n, node_out_features)

        self.trafo_mlp_o = nn.ModuleList(
            [
                nn.Linear(node_out_features, trafo_out_features, bias=False) for _ in range(trafo_hop + 1)
            ]
        )

        # self.trafo_mlp_a = nn.Linear(trafo_out_features, 1, bias = False) # output logits 
        self.trafo_mlp_a = nn.Linear(num_trafo_neighbors, 1, bias = False) # output logits 

        self.reset_parameters() 
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        # nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
        node_data, edge_data = tupleData[0], tupleData[1]
        data = node_data
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_data.x.to(self.device)
        edge_index_l = edge_data.edge_index.to(self.device)
        edge_weight_l = edge_data.edge_attr.to(self.device)
        edge_index_u = edge_data.edge_index_u.to(self.device)
        edge_weight_u = edge_data.edge_attr2.to(self.device)

        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                # x = F.relu(x)
                torch.tanh_(x)
            else: 
                x = layer(x)
        x_o = self.fc_node(x)

        # get the terminal buses of the specific trafo  
        trafo_edge = data.y_trafo_label[0][0]

        # since batch can have multiple graphs 
        batch_size = data.ptr.size(0)-1
        trafo_preds = [] # batch_size, logits 

        for i in range(batch_size):
            # tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
            # get node indices for the current graphg 
            start, end = data.ptr[i], data.ptr[i+1]
            curr_G_x = x_o[start:end] 
            

            # get edges for the current graph 
            mask = (data.edge_index[0] >= start) & (data.edge_index[0] < end)
            curr_G_edges = data.edge_index[:, mask] - start # since start will be nonzero for ptr > 0

            # build graph 
            curr_G = nx.Graph()
            curr_G_elist = curr_G_edges.T.tolist()
            curr_G.add_edges_from(curr_G_elist)

            # group nodes by shortest path length 
            hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
            for node in trafo_edge:  # Terminal nodes (trafo_edge)
                shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                
                for neighbor, hop in shortest_hops.items():
                    if hop > 0 and neighbor not in trafo_edge:
                        hop_group_dict_all[hop].append(neighbor)
                    else:
                        hop_group_dict_all[hop].append(neighbor)
            
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
            
            
            # agg and concat 
            hop_stack = []
            for h in range(self.trafo_hop + 1):
                if len(hop_group_dict_uniq[h]) > 0:
                    neighbor_feats = curr_G_x[hop_group_dict_uniq[h]]  # Select features for nodes at hop h
                    omega_x = self.trafo_mlp_o[h](neighbor_feats)  # Shared MLP
                    hop_stack.append(omega_x)

            # Concatenate and transform
            hop_concat = torch.cat(hop_stack, dim=0).t()  # Shape: (feature_dim, num_nodes)
            trafo_pred = self.trafo_mlp_a(hop_concat).t()  # Shape: (num_nodes, output_dim)
            
            trafo_preds.append(trafo_pred)
        
        # combine all keeping batch-wise structure 
        all_G_trafo_preds = torch.cat(trafo_preds, dim=0)
        
        return x_o, all_G_trafo_preds





# #########################################################################################################

# #########################################################################################################


class EdgeRegressor(nn.Module):
    def __init__(self, 
                 edge_input_features: int, 
                 list_edge_hidden_features: List[int],
                 edge_output_features: int,
                 k_hop_edge: int, 
                 bias: bool = True, 
                 normalize: bool = False, 
                 adj_norm: bool = True,
                 device: Literal['cuda','mps','cpu'] = 'cpu',
    ):
        super().__init__()
        self.name = "EdgeRegressor" # used in logging 
        self.bias = bias
        self.device = device

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = nn.ModuleList()
        self.edge_biases = nn.ParameterList()

        in_feats_e = edge_input_features

        # add bias to SCNN as a whole (rather than individual TAGConv above)
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features):

                self.edge_layers.append(nn.ModuleList([TAGConv(in_channels=in_feats_e, 
                                                out_channels=hid_feats_e,
                                                K=k_hop_edge,
                                                bias=False,
                                                normalize=False),
                                        # TAGConv(in_channels=in_feats_e,
                                        #         out_channels=hid_feats_e, 
                                        #         K=k_hop_edge,
                                        #         bias=False, # TODO: True? 
                                        #         normalize=adj_norm)
                                                ]))
                if bias:
                    self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
                else: 
                    self.edge_biases.append(None)
                in_feats_e = hid_feats_e
        else: 
            hid_feats_e = in_feats_e

        self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)


        self.reset_parameters() 
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        # nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_data = tupleData[0]

        edge_attr = edge_data.x.to(self.device)
        edge_index_l = edge_data.edge_index.to(self.device)
        edge_weight_l = edge_data.edge_attr.to(self.device)
        edge_index_u = edge_data.edge_index_u.to(self.device)
        edge_weight_u = edge_data.edge_attr2.to(self.device)
        
        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            edge_attr = layer[0](edge_attr, edge_index_l, edge_weight_l) #+ layer[1](edge_attr, edge_index_u, edge_weight_u)
            if self.bias: 
                edge_attr += e_bias
            torch.relu_(edge_attr) 

        edge_attr = self.fc_edge(edge_attr)
        
        return edge_attr 





# #########################################################################################################




class EdgeLGRegressor(nn.Module):
    def __init__(self, 
                 edge_input_features: int, 
                 list_edge_hidden_features: List[int],
                 edge_output_features: int,
                 k_hop_edge: int, 
                 bias: bool = True, 
                 normalize: bool = False, 
                 adj_norm: bool = True,
                 device: Literal['cuda','mps','cpu'] = 'cpu',
    ):
        super().__init__()
        self.name = "EdgeLGRegressor" # used in logging 
        self.bias = bias
        self.device = device

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = nn.ModuleList()
        self.edge_biases = nn.ParameterList()

        in_feats_e = edge_input_features

        # add bias to SCNN as a whole (rather than individual TAGConv above)
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features):

                self.edge_layers.append(TAGConv(in_channels=in_feats_e, 
                                                out_channels=hid_feats_e,
                                                K=k_hop_edge,
                                                bias=False,
                                                normalize=False))
                if bias:
                    self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
                else: 
                    self.edge_biases.append(None)
                in_feats_e = hid_feats_e
        else: 
            hid_feats_e = in_feats_e

        self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)


        self.reset_parameters() 
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        # if using relu
                        # nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        
                        # if using tanh 
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_lg_data = tupleData[1]
        edge_attr = edge_lg_data.x.to(self.device)
        # lower laplacian and upper laplacian
        edge_index_lg = edge_lg_data.edge_index.to(self.device)
        
        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            edge_attr = layer(edge_attr, edge_index_lg)
            if self.bias: 
                edge_attr += e_bias
            torch.relu_(edge_attr) 

        edge_attr = self.fc_edge(edge_attr)
        
        return edge_attr 

#########################################################################################################
#########################################################################################################


class NEGATRegressor_LGL(NEGATRegressor): 
    """Only SE with Node and Edge Regression followed by GATConv. Edge Regression uses Linegraph 
    Laplacian. 
    """
    def __init__(self, 
                 node_input_features: int, 
                 list_node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int,  
                 edge_input_features: int, 
                 edge_output_features: int, 
                 k_hop_edge: int, 
                 list_edge_hidden_features: list, 
                 gat_out_features: int, 
                 gat_head: int,
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True, # normalize the adjacency matrix (recommended)
                 device: Literal['cuda','cpu','mps'] = 'cpu',
    ):
        super().__init__(node_input_features=node_input_features,
                         list_node_hidden_features=list_node_hidden_features,
                         node_out_features=node_out_features,
                         k_hop_node=k_hop_node,
                         edge_input_features=edge_input_features,
                         edge_output_features=edge_output_features,
                         k_hop_edge=k_hop_edge,
                         list_edge_hidden_features=list_edge_hidden_features,
                         gat_out_features=gat_out_features,
                         gat_head=gat_head,
                         bias=bias,
                         normalize=normalize,
                         adj_norm=adj_norm,
                         device=device
                         )
        self.name = "NEGATRegressor_LGL"
        self.bias = bias 
        self.device = device 

        ###### SCNN: edge-regression convolution layers ######
        self.edge_layers = nn.ModuleList()
        self.edge_biases = nn.ParameterList()

        in_feats_e = edge_input_features

        # add bias to SCNN as a whole (rather than individual TAGConv above)
        if len(list_edge_hidden_features) != 0: 
            for idx, hid_feats_e in enumerate(list_edge_hidden_features):

                self.edge_layers.append(TAGConv(in_channels=in_feats_e, 
                                                out_channels=hid_feats_e,
                                                K=k_hop_edge,
                                                bias=False,
                                                normalize=False))
                if bias:
                    self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
                else: 
                    self.edge_biases.append(None)
                in_feats_e = hid_feats_e
        else: 
            hid_feats_e = in_feats_e

        self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

        super().reset_parameters() 

    def forward(self, tupleData) -> torch.Tensor: 
        node_data, edge_lg_data = tupleData[0], tupleData[2]
        x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
        x1 = edge_lg_data.x.to(self.device)
        # linegraph laplacian edge indices
        edge_index_lg = edge_lg_data.edge_index.to(self.device)

        # node-regression 
        for layer in self.node_layers:
            # no activation for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                x = F.relu(x)
                # torch.tanh_(x)
            else: 
                x = layer(x)

        x = self.fc_node(x)
        
        # edge-regression
        for layer, e_bias in zip(self.edge_layers, self.edge_biases):
            x1 = layer(x1, edge_index_lg) # this is where linegraph laplacian gets implemented. 
            if self.bias: 
                x1 += e_bias
            torch.relu_(x1) 

        x1 = self.fc_edge(x1)
        
        # gatconv
        alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

        # agg_mssgs to SE predictions
        x_o = self.mlp_gat(alpha_gat)

        return x_o





#########################################################################################################
#########################################################################################################




# #########################################################################################
# # Node Regressor # TAGConv
# #########################################################################################

class TAGNRegressor(nn.Module):
    def __init__(self, 
                 node_in_features: int, 
                 node_hidden_features: List[int], 
                 node_out_features: int, 
                 k_hop_node: int, # GNN filter order 
                 bias: bool = True, 
                 normalize: bool = True, 
                 adj_norm: bool = True # recommended.
    ):
        super().__init__()
        self.node_in = node_in_features
        self.node_hid = node_hidden_features
        self.node_out = node_out_features
        # self.num_node_layers = len(node_hidden_features)
        self.Kn = k_hop_node
        self.name = "TAGNRegressor"

        # node regression convolution layers 
        self.node_layers = nn.ModuleList()
        in_feats_n = node_in_features 

        if len(node_hidden_features) != 0: # if no GNN layers, directly use linear layer
            for idx, hid_feats_n in enumerate(node_hidden_features): 
                self.node_layers.append(TAGConv(in_channels=in_feats_n, 
                                                out_channels=hid_feats_n, 
                                                K=k_hop_node, 
                                                bias=bias, 
                                                normalize=adj_norm))
                # no normalization after last layer
                if normalize and idx < len(node_hidden_features): 
                    self.node_layers.append(LayerNorm(hid_feats_n))
                in_feats_n = hid_feats_n
        else: 
            hid_feats_n = in_feats_n
        
        self.fc = nn.Linear(hid_feats_n,node_out_features)
        self.reset_parameters()
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, MessagePassing):
                # using kaiming initialization for each layer in GNN 
                for name, param in layer.named_parameters(): # iterator over parameters 
                    if "weight" in name: 
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                        # different from layer.reset_parameters() --> uniform 
                        # suitable for relu 
                    elif "bias" in name and param is not None: # if bias = False 
                        nn.init.constant_(param, 0.1)
            elif isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, tupleData) -> Tuple[torch.Tensor]:
        data = tupleData[0]
        x, edge_index = data.x, data.edge_index # always ensure x, edge_index are same torch dtype 
        # node-regression 
        for layer in self.node_layers:
            # no relu for LayerNorm 
            if isinstance(layer, TAGConv): 
                x = layer(x, edge_index)
                # x = F.relu(x)
                torch.relu_(x)
            else: 
                x = layer(x)

        x = self.fc(x)
        return x 
    
class FCNNRegressor(nn.Module):
    def __init__(self, 
                 in_feat: int, 
                 hid_feat_list: List[int], 
                 out_feat: int,  
    ):
        super().__init__()
        self.node_in = in_feat
        self.node_hid = hid_feat_list
        self.node_out = out_feat
        self.name = "FCNNRegressor"

        layers = []

        last_feat = in_feat
        for hid_feat in hid_feat_list: 
            layers.append(nn.Linear(last_feat, hid_feat))
            layers.append(nn.ReLU())
            last_feat = hid_feat 
        
        layers.append(nn.Linear(last_feat, out_feat))
        
        self.all_layers = nn.Sequential(*layers) # * unpacks the list 
        self.reset_parameters()
    
    def reset_parameters(self):
        def reset_layer(layer): 
            """Helper function to avoid redundancy"""
            if isinstance(layer, nn.Linear): 
                # uniform_ means the operation modifies tensor in-place
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None: 
                    # zero initialize prevents learning angles 
                    nn.init.constant_(layer.bias, 0.01)
        self.apply(reset_layer)
    
    def forward(self, data) -> Tuple[torch.Tensor]:
        x = data
        return self.all_layers(x)
 


# class MultiTapSEGNN(nn.Module): 
#     """
#     Predicting multi/all tap-positions and performing SE using GATConv after GNN+SCNN.
    
#     This model is an extension of TapSEGNN. 
#     """
#     def __init__(self, 
#                  node_input_features: int, 
#                  list_node_hidden_features: List[int], 
#                  node_out_features: int, 
#                  k_hop_node: int, 
#                  edge_input_features: int, 
#                  list_edge_hidden_features: List[int],
#                  edge_output_features: int, 
#                  k_hop_edge: int, 
#                  trafo_hop: int, 
#                  num_trafo_neighbors: List, 
#                  trafo_out_features: List, # number of tap classes
#                  gat_out_features: int,  
#                  gat_head: int, 
#                  bias: bool = True, 
#                  normalize: bool = True, 
#                  adj_norm: bool = True, # adjacency not normalized for SCNN (returns NaNs)
#                  device: Literal['cuda','mps','cpu'] = 'cpu',
#                  ): 
#         super().__init__()
#         self.name = "MultiTapSEGNN"
#         self.bias = bias 
#         self.trafo_hop = trafo_hop
#         self.device = device

#         ###### GNN: node regression convolution layers ###### 
#         self.node_layers = nn.ModuleList()
#         in_feats_n = node_input_features 
         
#         if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
#             for idx, hid_feats_n in enumerate(list_node_hidden_features): 
#                 self.node_layers.append(TAGConv(in_channels=in_feats_n, 
#                                                 out_channels=hid_feats_n, 
#                                                 K=k_hop_node, 
#                                                 bias=bias, 
#                                                 normalize=adj_norm))
#                 # no normalization after last layer
#                 if normalize and idx < len(list_node_hidden_features): 
#                     self.node_layers.append(LayerNorm(hid_feats_n))
#                 in_feats_n = hid_feats_n
#         else: 
#             hid_feats_n = in_feats_n
        
#         self.fc_node = nn.Linear(hid_feats_n, node_out_features)

#         ###### SCNN: edge-regression convolution layers ######
#         self.edge_layers = nn.ModuleList()
#         self.edge_biases = nn.ParameterList()

#         in_feats_e = edge_input_features

#         # add bias to SCNN as a whole (rather than individual TAGConv above)
#         if len(list_edge_hidden_features) != 0: 
#             for idx, hid_feats_e in enumerate(list_edge_hidden_features):

#                 self.edge_layers.append(nn.ModuleList([TAGConv(in_channels=in_feats_e, 
#                                                 out_channels=hid_feats_e,
#                                                 K=k_hop_edge,
#                                                 bias=False,
#                                                 normalize=False),
#                                         TAGConv(in_channels=in_feats_e,
#                                                 out_channels=hid_feats_e, 
#                                                 K=k_hop_edge,
#                                                 bias=False, # TODO: True? 
#                                                 normalize=False)]))
#                 if bias:
#                     self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
#                 else: 
#                     self.edge_biases.append(None)
#                 in_feats_e = hid_feats_e
#         else: 
#             hid_feats_e = in_feats_e

#         self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

#         self.gatconv = GATConv(in_channels=node_out_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_output_features)

#         # self.gatconv = GATConv(in_channels=node_input_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_input_features)

#         # since gatconv with multiple heats concatenates outputs, a final regression layer is required. 
#         # mlp 
#         self.mlp_gat = nn.Sequential(
#             nn.Linear(gat_out_features * self.gatconv.heads, 2, bias = True),
#             nn.ReLU(), # TODO: Dropout required?
#         )

#         # TODO: Maybe use different parameters for each transformer???
#         # So that, trafo_mlp_o and trafo_mlp_a will be different for each transformer...
        
#         # note: trafo out features are the number of classes 
#         if all(feat == trafo_out_features[0] for feat in trafo_out_features):
#             print(f"All transformers have same number of taps, using shared parameter \omega.")

#             # consider all transformer have same number of taps  
#             self.multi_trafo_mlp_o = nn.ModuleList(
#                 [
#                     nn.Linear(2, int(trafo_out_features[0]), bias = False) for _ in range(trafo_hop + 1)
#                 ]
#             )
#         else: 
#             raise NotImplementedError("Not implemented for transformer with different number of tap classes.")
        
#         # self.trafo_mlp_a = nn.Linear(trafo_out_features, 1, bias = False) # output logits 

#         # since every transformer can have different number of neighbors 
#         self.multi_trafo_mlp_a = nn.ModuleDict() 
#         for trafo_id, num_trafo_neighbors_for_trafo_id in enumerate(num_trafo_neighbors):
#             self.multi_trafo_mlp_a["trafo a - "+str(trafo_id)] = nn.Linear(int(num_trafo_neighbors_for_trafo_id), 1, bias=False)


#         self.reset_parameters() 
    
#     def reset_parameters(self):
#         def reset_layer(layer): 
#             """Helper function to avoid redundancy"""
#             if isinstance(layer, MessagePassing):
#                 # using kaiming initialization for each layer in GNN 
#                 for name, param in layer.named_parameters(): # iterator over parameters 
#                     if "weight" in name: 
#                         # if using relu
#                         # nn.init.kaiming_uniform_(param, nonlinearity='relu')
#                         # different from layer.reset_parameters() --> uniform 
                        
#                         # if using tanh 
#                         nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

#                     elif "bias" in name and param is not None: # if bias = False 
#                         nn.init.constant_(param, 0.1)
#             elif isinstance(layer, nn.Linear): 
#                 # uniform_ means the operation modifies tensor in-place
#                 # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#                 nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
#                 if layer.bias is not None: 
#                     # zero initialize prevents learning angles 
#                     nn.init.constant_(layer.bias, 0.01)
#         self.apply(reset_layer)

#     def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
#         node_data, edge_data = tupleData[0], tupleData[1]
#         x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
#         x1 = edge_data.x.to(self.device)
        
#         edge_index_l = edge_data.edge_index.to(self.device)
#         edge_weight_l = edge_data.edge_attr.to(self.device)
#         edge_index_u = edge_data.edge_index_u.to(self.device)
#         edge_weight_u = edge_data.edge_attr2.to(self.device)
        
        
#         # node-regression 
#         for layer in self.node_layers:
#             # no activation for LayerNorm 
#             if isinstance(layer, TAGConv): 
#                 x = layer(x, edge_index)
#                 # x = F.relu(x)
#                 torch.relu_(x)
#             else: 
#                 x = layer(x)

#         x = self.fc_node(x)


#         # edge-regression
#         for layer, e_bias in zip(self.edge_layers, self.edge_biases):
#             x1 = layer[0](x=x1, edge_index=edge_index_l, edge_weight=edge_weight_l) 
#                 #   + layer[1](x=x1, edge_index=edge_index_u, edge_weight=edge_weight_u)
            
#             if self.bias: 
#                 x1 += e_bias
#             torch.relu_(x1) 

#         x1 = self.fc_edge(x1)
  
        
#         # gatconv
#         alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

#         # agg_mssgs to SE predictions
#         x_o = self.mlp_gat(alpha_gat)
        
#         multi_trafo_pred = dict()

#         # since the trafo_edges (hv_node, lv_node) will be the same for all the transformers across all the samples 
#         # in y_trafo_label select first sample to enumerate trafo_id and trafo_edges

#         for trafo_id, trafo_dict in enumerate(node_data.y_trafo_label[0]): 
            
#             # get terminal buses for trafo_id trafo 
#             trafo_edge = (trafo_dict["hv_node"], trafo_dict["lv_node"])
            
#             # since batch can have multiple graphs 
#             batch_size = node_data.ptr.size(0) -1 
            
#             single_trafo_preds = [] # batch_size, logits 

#             for i in range(batch_size):
#                 # tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
#                 # get node indices for the current graphg 
#                 start, end = node_data.ptr[i], node_data.ptr[i+1]
#                 curr_G_x = x_o[start:end] 
                
#                 # get edges for the current graph 
#                 mask = (node_data.edge_index[0] >= start) & (node_data.edge_index[0] < end)
#                 curr_G_edges = node_data.edge_index[:, mask] - start # since start will be nonzero for ptr > 0

#                 # build graph 
#                 curr_G = nx.Graph()
#                 curr_G_elist = curr_G_edges.T.tolist()
#                 curr_G.add_edges_from(curr_G_elist)

#                 # group nodes by shortest path length 
#                 hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
#                 for node in trafo_edge:  # Terminal nodes (trafo_edge)
#                     shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                    
#                     for neighbor, hop in shortest_hops.items():
#                         if hop > 0 and neighbor not in trafo_edge:
#                             hop_group_dict_all[hop].append(neighbor)
#                         else:
#                             hop_group_dict_all[hop].append(neighbor)
                
#                 # only consider new and unique nodes for next hop starting with terminal nodes for 0 hop 
#                 hop_group_dict_uniq = {key: [] for key in hop_group_dict_all.keys()}
#                 all_uniq_nodes = set() # unique nodes considered for all hops 

#                 for key in hop_group_dict_all.keys():
#                     hop_uniq_nodes = set(hop_group_dict_all[key]) # no repeatition 
#                     # if uniq nodes at hop are not already in all_uniq nodes
#                     if not all_uniq_nodes & hop_uniq_nodes: 
#                         all_uniq_nodes.update(hop_group_dict_all[key])
#                         hop_group_dict_uniq[key] = hop_group_dict_all[key]
#                     else: 
#                         for and_node in list(all_uniq_nodes & hop_uniq_nodes): 
#                             hop_uniq_nodes.remove(and_node)
#                         all_uniq_nodes.update(hop_uniq_nodes)
#                         hop_group_dict_uniq[key] = list(hop_uniq_nodes)
                
#                 # agg and concat 
#                 hop_stack = []
#                 for h in range(self.trafo_hop + 1):
#                     if len(hop_group_dict_uniq[h]) > 0:
#                         neighbor_feats = curr_G_x[hop_group_dict_uniq[h]]  # Select features for nodes at hop h
#                         omega_x = self.multi_trafo_mlp_o[h](neighbor_feats)  # Shared parameters 
#                         hop_stack.append(omega_x)

#                 # Concatenate and transform
#                 hop_concat = torch.cat(hop_stack, dim=0).t()  # Shape: (feature_dim, num_nodes)
#                 trafo_pred = self.multi_trafo_mlp_a["trafo a - "+str(trafo_id)](hop_concat).t()  # Shape: (num_nodes, output_dim)
                
#                 single_trafo_preds.append(trafo_pred)
            
#             # combine all keeping batch-wise structure 
#             all_G_trafo_preds = torch.cat(single_trafo_preds, dim=0)
        
#             multi_trafo_pred[trafo_id] = all_G_trafo_preds

#         return x_o, multi_trafo_pred


####################################################################################



# class IntegratedGNN(nn.Module):
#     def __init__(self, 
#                  node_in_features: int, 
#                  node_hidden_features: int, 
#                  node_out_features: int, 
#                  k_hop_node: int, # GNN filter order 
#                  num_node_layers: int, 
#                  edge_in_features: int, 
#                  edge_hidden_features: int, 
#                  edge_out_features: int, 
#                  k_hop_edge: int, 
#                  num_edge_layers: int, 
#                 # any_triangles: bool = True, 
#                  mlp_out: int, 
#                  mlp_omega_out: int,
#                  bias: bool = True, 
#                  normalize: bool = True
#     ):
#         super().__init__()
#         self.node_in = node_in_features
#         self.node_hid = node_hidden_features
#         self.node_out = node_out_features
#         self.num_node_layers = num_node_layers
#         self.Kn = k_hop_node
#         self.Ke = k_hop_edge
#         self.edge_in = edge_in_features
#         self.edge_hid = edge_hidden_features
#         self.edge_out = edge_out_features
#         self.num_edge_layers = num_edge_layers
#         self.short_name = "IGNN"
        
#         # node regression convolution layers 
#         self.node_layers = nn.ModuleList()
#         in_feats_n = node_in_features 
#         for num in range(num_node_layers):
#             out_feats_n = node_out_features if num == num_node_layers - 1 else node_hidden_features 
#             self.node_layers.append(TAGConv(in_channels=in_feats_n, 
#                                             out_channels=out_feats_n, 
#                                             K=k_hop_node, 
#                                             bias=bias, 
#                                             normalize=normalize))
#             in_feats_n = node_hidden_features

#         # edge regression layers 
#         self.edge_bias = False 
#         self.edge_layers = nn.ModuleList()
#         for num in range(num_edge_layers):
#             in_feats_e = edge_in_features if num == 0 else edge_hidden_features
#             out_feats_e = edge_out_features if num == num_edge_layers - 1 else edge_hidden_features

#             self.edge_layers.append(nn.ModuleList(
#                 [
#                     TAGConv(in_channels=in_feats_e, out_channels=out_feats_e, K=k_hop_edge, bias=self.edge_bias, normalize=normalize), 
#                     TAGConv(in_channels=in_feats_e, out_channels=out_feats_e, K=k_hop_edge, bias=self.edge_bias, normalize=normalize)
#                 ]
#             ))
        
#         # Due to the structure of the simplicial complex neural networks 
#         self.he_biases = []
#         for layer in range(num_edge_layers):
#             if bias: 
#                 layer_bias =nn.Parameter(
#                     torch.Tensor(edge_hidden_features if layer < num_edge_layers - 1 else edge_out_features)
#                 )
#                 self.he_biases.append(layer_bias)
#         self.he_biases = nn.ParameterList(self.he_biases)

#         # MLP: Eq 5-3 
#         self.mlp_agg = nn.Sequential(
#             nn.Linear(2*out_feats_n + out_feats_e, mlp_out, bias = True), 
#             nn.ReLU(), 
#             nn.Dropout(0.5),
#         )
        
#         # MLP: part of Eq 5-6
#         self.mlp_omega = nn.Sequential(
#             nn.Linear(mlp_out, mlp_omega_out, bias = False),
#             nn.ReLU()
#         )
        

#         # MLP: rest of Eq 5-6
#         self.mlp_a = nn.Sequential(
#             nn.Linear(2*mlp_omega_out, mlp_omega_out, bias = True), 
#             nn.ReLU()
#         )

#         self.reset_parameters()
    
#     def reset_parameters(self):
#         def reset_layer(layer): 
#             """Helper function to avoid redundancy"""
#             if isinstance(layer, MessagePassing):
#                 # using kaiming initialization for each layer in GNN 
#                 for name, param in layer.named_parameters(): # iterator over parameters 
#                     if "weight" in name: 
#                         nn.init.kaiming_uniform_(param, nonlinearity='relu')
#                         # different from layer.reset_parameters() --> uniform 
#                         # suitable for relu 
#                     elif "bias" in name and param is not None: # if bias = False 
#                         nn.init.constant_(param, 0.1)
#             elif isinstance(layer, nn.Linear): 
#                 # uniform_ means the operation modifies tensor in-place
#                 nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#                 if layer.bias is not None: 
#                     # zero initialize prevents learning angles 
#                     nn.init.constant_(layer.bias, 0.01)
#         self.apply(reset_layer)

    
#     def sparse_adj_power(self, sparse_adj, power):
#         if power == 0: # then return identity matrix
#             sp_ident = torch.eye(sparse_adj.shape[0])
#             return sp_ident.to_sparse()
#         result = sparse_adj
#         for _ in range(1,power):
#             result = torch.sparse.mm(result, sparse_adj)
#         return result
    
#     def sum_concat(self, x: torch.Tensor, edge_index:torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
#         # abiding by notations in the thesis where cap is capital 
#         cap_y_hat = x.clone() 
#         cap_y_dash_hat = edge_attr.clone()

#         # [num_nodes, 2*node_embeddings + edge_embeddings(connecting)]
#         agg_mssgs = torch.zeros((cap_y_hat.shape[0], 2*cap_y_hat.shape[1] + cap_y_dash_hat.shape[1]))

#         # agg_mssgs = [node_i_features, node_j_features, edge_k_features]
#         for e in range(edge_index.shape[1]):
#             i, j = edge_index[:,e] 
#             y_hat_i, y_hat_j = x[i], x[j] 
#             edge_ij = edge_attr[e]

#             agg_mssgs[i] += torch.cat([y_hat_i, y_hat_j, edge_ij], dim=-1) # row vectors
#             agg_mssgs[j] += torch.cat([y_hat_j, y_hat_i, edge_ij], dim=-1)

#         return agg_mssgs

    
#     def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         # node-regression 
#         for conv in self.node_layers:
#             x = conv(x, edge_index) # not considering any edge-attributes here
#             x = torch.relu(x) 
        
#         # node-regression output [:,node_out_features]

#         # edge-regression
#         hodge_L, any_triangles = construct_hodge_laplacian(data, dense = False)
#         if any_triangles: 
#             coo_l = hodge_L[0].tocoo()
#             edge_index_L_l = torch.tensor(np.vstack((coo_l.row, coo_l.col)), dtype = torch.long)
#             coo_u = hodge_L[1].tocoo()
#             edge_index_L_u = torch.tensor(np.vstack((coo_u.row, coo_u.col)), dtype = torch.long)
#             for layer, conv_pair in enumerate(self.edge_layers):
#                 edge_attr = conv_pair[0](edge_attr, edge_index_L_l) + conv_pair[1](edge_attr, edge_index_L_u) # hodge_L is the laplacian matrix
#                 if layer < self.num_edge_layers - 1:
#                     edge_attr += self.he_biases[layer]
#                 edge_attr = torch.relu(edge_attr)
#         else: 
#             coo_l = hodge_L[0].tocoo()
#             edge_index_L_l = torch.tensor(np.vstack((coo_l.row, coo_l.col)), dtype = torch.long)
#             for layer, conv_pair in enumerate(self.edge_layers):
#                 edge_attr = conv_pair[0](edge_attr, edge_index_L_l)
#                 if layer < self.num_edge_layers - 1:
#                     edge_attr += self.he_biases[layer]
#                 edge_attr = torch.relu(edge_attr)
        
#         agg_mssgs = self.sum_concat(x, edge_index, edge_attr)

#         # implement MLP (dropout not proposed in the methodology)
#         x_o = self.mlp_agg(agg_mssgs)

#         # trafo_mask bool  
#         trafo_mask = data.trafo_mask # caligraphic N_t
        
#         trafo_edge_index = edge_index[:, trafo_mask]

#         t_ij = torch.zeros((sum(trafo_mask)))

#         for e in range(sum(trafo_mask)): 
#             i, j = trafo_edge_index[:,e]
#             x_o_i, x_o_j = x_o[i], x_o[j]

#             # equation 5-6, with no bias
#             x_hat_i = self.mlp_omega(x_o_i)
#             x_hat_j = self.mlp_omega(x_o_j)

#             cat_x_hat = torch.cat([x_hat_i, x_hat_j], dim=-1)

#             # with bias 
#             t_ij[e] = self.mlp_a(cat_x_hat)

#         return x_o, t_ij.shape

##################################################################################################################################

# class TapSEGNN(nn.Module): 
#     """Predicting single tap-positions and performing SE using GATConv after GNN+SCNN."""
#     def __init__(self,
#                  node_input_features: int, 
#                  list_node_hidden_features: List[int], 
#                  node_out_features: int, 
#                  k_hop_node: int,  
#                  edge_input_features: int, 
#                  list_edge_hidden_features: List[int],
#                  edge_output_features: int, 
#                  k_hop_edge: int, 
#                  trafo_hop: int,
#                  num_trafo_neighbors: int, 
#                  trafo_out_features: int, # number of tap classes
#                  gat_out_features: int, 
#                  gat_head: int,
#                  bias: bool = True, 
#                  normalize: bool = True, 
#                  adj_norm: bool = True, # normalize the adjacency matrix (recommended)
#                  device: Literal['cuda','mps','cpu'] = 'cpu',
#                  ):
#         super().__init__()
#         self.name = "TapSEGNN" # used in logging 
#         self.bias = bias
#         self.trafo_hop = trafo_hop
#         self.device = device

#         ###### GNN: node regression convolution layers ###### 
#         self.node_layers = nn.ModuleList()
#         in_feats_n = node_input_features 
         
#         if len(list_node_hidden_features) != 0: # if no GNN layers, directly use linear layer
#             for idx, hid_feats_n in enumerate(list_node_hidden_features): 
#                 self.node_layers.append(TAGConv(in_channels=in_feats_n, 
#                                                 out_channels=hid_feats_n, 
#                                                 K=k_hop_node, 
#                                                 bias=bias, 
#                                                 normalize=adj_norm))
#                 # no normalization after last layer
#                 if normalize and idx < len(list_node_hidden_features): 
#                     self.node_layers.append(LayerNorm(hid_feats_n))
#                 in_feats_n = hid_feats_n
#         else: 
#             hid_feats_n = in_feats_n
        
#         self.fc_node = nn.Linear(hid_feats_n, node_out_features)

#         ###### SCNN: edge-regression convolution layers ######
#         self.edge_layers = nn.ModuleList()
#         self.edge_biases = nn.ParameterList()

#         in_feats_e = edge_input_features

#         # add bias to SCNN as a whole (rather than individual TAGConv above)
#         if len(list_edge_hidden_features) != 0: 
#             for idx, hid_feats_e in enumerate(list_edge_hidden_features):

#                 self.edge_layers.append(nn.ModuleList([TAGConv(in_channels=in_feats_e, 
#                                                 out_channels=hid_feats_e,
#                                                 K=k_hop_edge,
#                                                 bias=False,
#                                                 normalize=False),
#                                         TAGConv(in_channels=in_feats_e,
#                                                 out_channels=hid_feats_e, 
#                                                 K=k_hop_edge,
#                                                 bias=False, # TODO: True? 
#                                                 normalize=False)]))
#                 if bias:
#                     self.edge_biases.append(nn.Parameter(torch.Tensor(hid_feats_e)))
#                 else: 
#                     self.edge_biases.append(None)
#                 in_feats_e = hid_feats_e
#         else: 
#             hid_feats_e = in_feats_e

#         self.fc_edge = nn.Linear(hid_feats_e, edge_output_features)

#         self.gatconv = GATConv(in_channels=node_out_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_output_features)

#         # self.gatconv = GATConv(in_channels=node_input_features, out_channels=gat_out_features, heads=gat_head, edge_dim=edge_input_features)

#         # since gatconv with multiple heats concatenates outputs, a final regression layer is required. 
#         # mlp 
#         self.mlp_gat = nn.Sequential(
#             nn.Linear(gat_out_features * self.gatconv.heads, 2, bias = True),
#             # nn.ReLU(), # TODO: Dropout required?
#         )

#         # TODO: Maybe use different parameters for each transformer???
#         # So that, trafo_mlp_o and trafo_mlp_a will be different for each transformer...

#         self.trafo_mlp_o = nn.ModuleList(
#             [
#                 nn.Linear(2, trafo_out_features, bias=False) for _ in range(trafo_hop + 1)
#             ]
#         )

#         # self.trafo_mlp_a = nn.Linear(trafo_out_features, 1, bias = False) # output logits 
#         self.trafo_mlp_a = nn.Linear(num_trafo_neighbors, 1, bias = False) # output logits 

#         self.reset_parameters() 
    
#     def reset_parameters(self):
#         def reset_layer(layer): 
#             """Helper function to avoid redundancy"""
#             if isinstance(layer, MessagePassing):
#                 # using kaiming initialization for each layer in GNN 
#                 for name, param in layer.named_parameters(): # iterator over parameters 
#                     if "weight" in name: 
#                         # if using relu
#                         # nn.init.kaiming_uniform_(param, nonlinearity='relu')
#                         # different from layer.reset_parameters() --> uniform 
                        
#                         # if using tanh 
#                         nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))

#                     elif "bias" in name and param is not None: # if bias = False 
#                         nn.init.constant_(param, 0.1)
#             elif isinstance(layer, nn.Linear): 
#                 # uniform_ means the operation modifies tensor in-place
#                 # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#                 nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
#                 if layer.bias is not None: 
#                     # zero initialize prevents learning angles 
#                     nn.init.constant_(layer.bias, 0.01)
#         self.apply(reset_layer)

#     def forward(self, tupleData) -> Tuple[torch.Tensor, torch.Tensor]:
#         node_data, edge_data = tupleData[0], tupleData[1]
#         x, edge_index = node_data.x.to(self.device), node_data.edge_index.to(self.device)
#         x1 = edge_data.x.to(self.device)
        
#         edge_index_l = edge_data.edge_index.to(self.device)
#         edge_weight_l = edge_data.edge_attr.to(self.device)
#         edge_index_u = edge_data.edge_index_u.to(self.device)
#         edge_weight_u = edge_data.edge_attr2.to(self.device)
        

#         # node-regression 
#         for layer in self.node_layers:
#             # no activation for LayerNorm 
#             if isinstance(layer, TAGConv): 
#                 x = layer(x, edge_index)
#                 # x = F.relu(x)
#                 torch.relu_(x)
#             else: 
#                 x = layer(x)

#         x = self.fc_node(x)


#         # edge-regression
#         for layer, e_bias in zip(self.edge_layers, self.edge_biases):
#             x1 = layer[0](x=x1, edge_index=edge_index_l, edge_weight=edge_weight_l) 
#                 #   + layer[1](x=x1, edge_index=edge_index_u, edge_weight=edge_weight_u)
            
#             if self.bias: 
#                 x1 += e_bias
#             torch.relu_(x1) 

#         x1 = self.fc_edge(x1)
  
        
#         # gatconv
#         alpha_gat = self.gatconv(x=x, edge_index=edge_index, edge_attr=x1)

#         # agg_mssgs to SE predictions
#         x_o = self.mlp_gat(alpha_gat)
        

#         # get the terminal buses of the specific trafo  
#         trafo_edge = node_data.y_trafo_label[0][0]

#         # since batch can have multiple graphs 
#         batch_size = node_data.ptr.size(0)-1
#         trafo_preds = [] # batch_size, logits 

#         for i in range(batch_size):
#             # tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
#             # get node indices for the current graphg 
#             start, end = node_data.ptr[i], node_data.ptr[i+1]
#             curr_G_x = x_o[start:end] 
            
#             # get edges for the current graph 
#             mask = (node_data.edge_index[0] >= start) & (node_data.edge_index[0] < end)
#             curr_G_edges = node_data.edge_index[:, mask] - start # since start will be nonzero for ptr > 0

#             # build graph 
#             curr_G = nx.Graph()
#             curr_G_elist = curr_G_edges.T.tolist()
#             curr_G.add_edges_from(curr_G_elist)

#             # group nodes by shortest path length 
#             hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
#             for node in trafo_edge:  # Terminal nodes (trafo_edge)
#                 shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                
#                 for neighbor, hop in shortest_hops.items():
#                     if hop > 0 and neighbor not in trafo_edge:
#                         hop_group_dict_all[hop].append(neighbor)
#                     else:
#                         hop_group_dict_all[hop].append(neighbor)
            
#             # only consider new and unique nodes for next hop starting with terminal nodes for 0 hop 
#             hop_group_dict_uniq = {key: [] for key in hop_group_dict_all.keys()}
#             all_uniq_nodes = set() # unique nodes considered for all hops 

#             for key in hop_group_dict_all.keys():
#                 hop_uniq_nodes = set(hop_group_dict_all[key]) # no repeatition 
#                 # if uniq nodes at hop are not already in all_uniq nodes
#                 if not all_uniq_nodes & hop_uniq_nodes: 
#                     all_uniq_nodes.update(hop_group_dict_all[key])
#                     hop_group_dict_uniq[key] = hop_group_dict_all[key]
#                 else: 
#                     for and_node in list(all_uniq_nodes & hop_uniq_nodes): 
#                         hop_uniq_nodes.remove(and_node)
#                     all_uniq_nodes.update(hop_uniq_nodes)
#                     hop_group_dict_uniq[key] = list(hop_uniq_nodes)
            
#             # agg and concat 
#             hop_stack = []
#             for h in range(self.trafo_hop + 1):
#                 if len(hop_group_dict_uniq[h]) > 0:
#                     neighbor_feats = curr_G_x[hop_group_dict_uniq[h]]  # Select features for nodes at hop h
#                     omega_x = self.trafo_mlp_o[h](neighbor_feats)  # Shared MLP
#                     hop_stack.append(omega_x)

#             # Concatenate and transform
#             hop_concat = torch.cat(hop_stack, dim=0).t()  # Shape: (feature_dim, num_nodes)
#             trafo_pred = self.trafo_mlp_a(hop_concat).t()  # Shape: (num_nodes, output_dim)
            
#             trafo_preds.append(trafo_pred)
        
#         # combine all keeping batch-wise structure 
#         all_G_trafo_preds = torch.cat(trafo_preds, dim=0)

#         return x_o, all_G_trafo_preds

#########################################################################################################

