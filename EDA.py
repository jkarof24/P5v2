import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def load_swapgt_benchmark(file_path):
    raw_content = torch.load(file_path, weights_only=False)
    
    if isinstance(raw_content, list) and len(raw_content) >= 3:
        adj_matrix = raw_content[0]  # Adjacency matrix (stored sparse or dense)
        x = raw_content[1]           # Node features
        y = raw_content[2]           # Labels
        
        # Handle sparse vs dense adjacency matrices cleanly
        if torch.is_tensor(adj_matrix):
            if adj_matrix.is_sparse:
                adj_matrix = adj_matrix.to_dense()
            edge_index, _ = dense_to_sparse(adj_matrix)
        else:
            edge_index = None

        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Attach masks safely if present
        if len(raw_content) >= 6:
            data.train_mask = torch.tensor(raw_content[3]) if not torch.is_tensor(raw_content[3]) else raw_content[3]
            data.val_mask = torch.tensor(raw_content[4]) if not torch.is_tensor(raw_content[4]) else raw_content[4]
            data.test_mask = torch.tensor(raw_content[5]) if not torch.is_tensor(raw_content[5]) else raw_content[5]
            
        return data
    return None

def scan_all_datasets():
    dataset_dir = os.path.join("SwapGT-data", "dataset")
    
    print("=" * 80)
    print(" SWAPGT BENCHMARK DATASET INSPECTION (CORRECTED LAYOUT)")
    print("=" * 80)

    for subset in sorted(os.listdir(dataset_dir)):
        subset_path = os.path.join(dataset_dir, subset)
        if not os.path.isdir(subset_path):
            continue
            
        print(f"\n--- Subfolder: [{subset.upper()}] ---")
        for filename in sorted(os.listdir(subset_path)):
            if filename.endswith('.pt'):
                file_path = os.path.join(subset_path, filename)
                try:
                    data = load_swapgt_benchmark(file_path)
                    if data:
                        num_classes = int(data.y.max().item() + 1) if data.y is not None else "N/A"
                        train_nodes = data.train_mask.sum().item() if hasattr(data, 'train_mask') else "N/A"
                        
                        print(f"  • {filename:<15} | Nodes: {data.num_nodes:<6} | Edges: {data.num_edges:<8} | Features: {data.num_node_features:<5} | Classes: {num_classes:<2} | Train Set: {train_nodes}")
                except Exception as e:
                    print(f"  • {filename:<15} | Error: {e}")
                    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    scan_all_datasets()