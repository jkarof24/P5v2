import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

def preprocess_datasets():
    src_dir = os.path.join("SwapGT-data", "dataset")
    out_dir = "clean_data"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print(" 🧹 STANDARDIZING & CLEANING DATASETS")
    print("=" * 60)
    
    for subset in ['de', 'sp']:
        sub_in = os.path.join(src_dir, subset)
        sub_out = os.path.join(out_dir, subset)
        os.makedirs(sub_out, exist_ok=True)
        
        if not os.path.exists(sub_in):
            continue
            
        for filename in sorted(os.listdir(sub_in)):
            if not filename.endswith('.pt'):
                continue
                
            file_path = os.path.join(sub_in, filename)
            raw_content = torch.load(file_path, weights_only=False)
            
            # Unpack adjacency
            adj = raw_content[0]
            if torch.is_tensor(adj):
                if adj.is_sparse:
                    adj = adj.to_dense()
                edge_index, _ = dense_to_sparse(adj)
            else:
                edge_index = None
                
            x = raw_content[1]
            y = raw_content[2]
            num_nodes = x.shape[0]
            
            data = Data(x=x, edge_index=edge_index, y=y)
            
            # Standardize masks
            for i, mask_name in enumerate(['train_mask', 'val_mask', 'test_mask'], start=3):
                if i < len(raw_content) and raw_content[i] is not None:
                    mask = raw_content[i]
                    if not torch.is_tensor(mask):
                        mask = torch.tensor(mask)
                    if mask.ndim > 1:
                        mask = mask[:, 0]
                        
                    if mask.dtype == torch.bool and mask.shape[0] == num_nodes:
                        setattr(data, mask_name, mask)
                    else:
                        bool_mask = torch.zeros(num_nodes, dtype=torch.bool)
                        bool_mask[mask.long()] = True
                        setattr(data, mask_name, bool_mask)
                        
            # Save standardized data object
            torch.save(data, os.path.join(sub_out, filename))
            print(f"  • Processed & Saved: [{subset.upper()}] {filename}")
            
    print("=" * 60)
    print("✅ Preprocessing complete! Clean data saved in 'clean_data/' folder.")

if __name__ == "__main__":
    preprocess_datasets()