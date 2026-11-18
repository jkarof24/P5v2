import os
from typing import List, Optional
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, degree, to_networkx

def _compute_node_degrees(edge_index: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Computes node degrees as a pure structural feature."""
    if edge_index is not None and edge_index.numel() > 0:
        row, _ = edge_index
        return degree(row, num_nodes=num_nodes, dtype=torch.float).unsqueeze(1)
    return torch.zeros((num_nodes, 1), dtype=torch.float)

def _compute_clustering_coefficient(edge_index: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Computes the local clustering coefficient for each node."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 1), dtype=torch.float)
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=True)
    g.add_nodes_from(range(num_nodes))  # Ensure isolated nodes are included
    
    clustering_dict = nx.clustering(g)
    clustering_list = [clustering_dict.get(i, 0.0) for i in range(num_nodes)]
    return torch.tensor(clustering_list, dtype=torch.float).unsqueeze(1)

def _compute_page_rank(edge_index: Optional[torch.Tensor], num_nodes: int, alpha: float = 0.85) -> torch.Tensor:
    """Computes the PageRank centrality score for each node."""
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 1), dtype=torch.float)
    
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    g = to_networkx(data, to_undirected=False)
    g.add_nodes_from(range(num_nodes))  # Ensure isolated nodes are included
    
    try:
        pr_dict = nx.pagerank(g, alpha=alpha)
    except Exception:
        pr_dict = {i: 1.0 / num_nodes for i in range(num_nodes)}
        
    pr_list = [pr_dict.get(i, 0.0) for i in range(num_nodes)]
    return torch.tensor(pr_list, dtype=torch.float).unsqueeze(1)

def node_features(edge_index: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """
    Modular feature pipeline combining node degrees, clustering coefficients, and PageRank.
    """
    features: List[torch.Tensor] = [
        _compute_node_degrees(edge_index, num_nodes),
        _compute_clustering_coefficient(edge_index, num_nodes),
        _compute_page_rank(edge_index, num_nodes),
    ]
    return torch.cat(features, dim=1)

def _standardize_mask(mask: Optional[torch.Tensor], num_nodes: int) -> Optional[torch.Tensor]:
    """Standardizes a mask into a boolean tensor of shape (num_nodes,)."""
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask)
    if mask.ndim > 1:
        mask = mask[:, 0]
        
    if mask.dtype == torch.bool and mask.shape[0] == num_nodes:
        return mask
        
    bool_mask = torch.zeros(num_nodes, dtype=torch.bool)
    bool_mask[mask.long()] = True
    return bool_mask

def process_graph(raw_content: tuple) -> Data:
    """Parses raw content, builds structural features, and packages into a PyG Data object."""
    adj, _, y, *masks = raw_content
    num_nodes = y.shape[0]

    # Convert adjacency matrix to edge_index
    edge_index = None
    if torch.is_tensor(adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        edge_index, _ = dense_to_sparse(adj)

    # Generate multi-feature node tensor
    x = node_features(edge_index, num_nodes)
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Standardize and attach masks
    mask_names = ['train_mask', 'val_mask', 'test_mask']
    for name, mask in zip(mask_names, masks):
        std_mask = _standardize_mask(mask, num_nodes)
        if std_mask is not None:
            setattr(data, name, std_mask)
            
    return data

def run(src_root: str = os.path.join("SwapGT-data", "dataset"), out_root: str = "clean_pure_data") -> None:
    """Drives the directory traversal, processing, and saving."""
    os.makedirs(out_root, exist_ok=True)
    
    for subset in ['de', 'sp']:
        sub_in = os.path.join(src_root, subset)
        sub_out = os.path.join(out_root, subset)
        
        if not os.path.exists(sub_in):
            continue
            
        os.makedirs(sub_out, exist_ok=True)
        for filename in sorted(os.listdir(sub_in)):
            if not filename.endswith('.pt'):
                continue
                
            file_path = os.path.join(sub_in, filename)
            raw_content = torch.load(file_path, weights_only=False)
            
            data = process_graph(raw_content)
            
            torch.save(data, os.path.join(sub_out, filename))
            print(f"  • Processed Pure-Structure: [{subset.upper()}] {filename}")

    print("done")

if __name__ == "__main__":
    run()