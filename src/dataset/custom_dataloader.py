import torch 
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
import networkx as nx 
from typing import Optional, Callable, List, Dict, Any 

class TransformerHopDataLoader(DataLoader):
    """
    Custom DataLoader that extends PyTorch's DataLoader to incorporate transformer hop 
    neighborhood calculations for each batch during loading.
    
    This class performs the transformer hop calculations as part of the data loading process,
    adding transformer hop neighborhood information to each batch.
    """

    def __init__(
            self, 
            dataset, 
            trafo_hop: int, 
            batch_size: int = 1, 
            shuffle: bool = False, 
            drop_last: bool = False, 
            **kwargs
    ): 
        """
        Initialize the TransformerHopDataLoader.
        
        Args:
            dataset: The dataset to load data from
            trafo_hop: Number of hops to consider for transformer neighborhood
            batch_size: Number of samples in each batch
            shuffle: Whether to shuffle the dataset
            drop_last: Whether to drop the last incomplete batch
            **kwargs: Additional arguments to pass to the parent DataLoader
        """
        self.trafo_hop = trafo_hop 
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last, 
            collate_fn=self._collate_with_trafo_hop,
            **kwargs
        )
        

    def _collate_with_trafo_hop(self, data_list):
        """
        Custom collate function that first creates a batch using PyTorch Geometric's Batch class,
        then calculates transformer hop neighborhoods for the batch.
        
        Args:
            data_list: List of data objects to collate
            
        Returns:
            batch: Processed batch with transformer hop neighborhood information
        """

        # Use PyTorch Geometric's Batch class to collate graph data
        batch_list = Batch.from_data_list(data_list)
        
        # Process the batch to add transformer hop neighborhood information
        self._process_trafo_hop_neighbors(batch_list)

        return batch_list 
    
    def _process_trafo_hop_neighbors(self, batch_list):
        """
        Calculate transformer hop neighbors for each graph in the batch.
        
        Args:
            batch: The batch of data to process
        """
        batch = batch_list[0]
        print(batch)
        exit()

        if not hasattr(batch, 'y_trafo_label') or batch.y_trafo_label is None:
            raise ValueError("Batch must have 'y_trafo_label' attribute for transformer hop calculations")
        
        # Get the terminal buses of the specific transformer
        trafo_edge = batch.y_trafo_label[0][0]
        
        # Calculate number of graphs in the batch
        num_graphs = int(len(batch.ptr) - 1)
        
        # Dictionary to store all unique neighbors by hop for the entire batch
        batch_hop_neighbors = {h: [] for h in range(self.trafo_hop + 1)}
        
        # Process each graph in the batch
        for i in range(num_graphs):
            # Get node indices for the current graph
            start, end = batch.ptr[i], batch.ptr[i+1]
            
            # Filter edges for the current graph
            mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
            curr_G_edges = batch.edge_index[:, mask] - start
            
            # Build NetworkX graph
            curr_G = nx.Graph()
            curr_G_elist = curr_G_edges.T.tolist()
            curr_G.add_edges_from(curr_G_elist)
            
            # Initialize hop group dictionaries
            hop_group_dict_all = {h: [] for h in range(self.trafo_hop + 1)}
            
            # For each terminal node of the transformer
            for node in trafo_edge:
                # Calculate shortest paths up to trafo_hop
                shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=self.trafo_hop)
                
                # Group nodes by hop distance
                for neighbor, hop in shortest_hops.items():
                    neighbor_idx = int(neighbor + start)
                    hop_group_dict_all[hop].append(neighbor_idx)
            
            # Process to get unique nodes for each hop
            hop_group_dict_uniq = self._get_unique_hop_groups(hop_group_dict_all)
            
            # Update batch hop neighbors
            for hop, nodes in hop_group_dict_uniq.items():
                batch_hop_neighbors[hop].extend(nodes)
        
        # Store the transformer hop neighborhood information in the batch
        batch.trafo_hop_uniq_neighbors = batch_hop_neighbors
        
        batch_list[0] = batch

        return batch_list
    
    def _get_unique_hop_groups(self, hop_group_dict_all):
        """
        Process hop groups to ensure nodes are only counted in their closest hop.
        
        Args:
            hop_group_dict_all: Dictionary mapping hop distances to lists of nodes
            
        Returns:
            Dictionary with unique node assignments to hop distances
        """
        hop_group_dict_uniq = {key: [] for key in hop_group_dict_all.keys()}
        all_uniq_nodes = set()  # Track unique nodes across all hops
        
        # Process hops in order (0, 1, 2, ...)
        for key in sorted(hop_group_dict_all.keys()):
            hop_uniq_nodes = set(hop_group_dict_all[key])  # Remove duplicates within hop
            
            # Find nodes not already included in previous hops
            new_nodes = hop_uniq_nodes - all_uniq_nodes
            
            # Update the set of all unique nodes and store new nodes for this hop
            all_uniq_nodes.update(new_nodes)
            hop_group_dict_uniq[key] = list(new_nodes)
            
        return hop_group_dict_uniq
    

    