import os
import torch
import torch.nn.functional as F
from model import XGBoostModel, GCN, GraphSAGE, GCNII, PolyFormer, SwapGTModel

def run_benchmark():
    clean_dir = "clean_pure_data"

    models_to_bench = {
        "XGBoost": XGBoostModel,
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "GCNII": GCNII,
        "PolyFormer": PolyFormer,
        "SwapGT": SwapGTModel
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    for model_name, model_class in models_to_bench.items():
        print(f"[{model_name}]")
        
        for subset in ['de', 'sp']:
            sub_path = os.path.join(clean_dir, subset)
            if not os.path.exists(sub_path):
                continue
                
            for filename in sorted(os.listdir(sub_path)):
                if not filename.endswith('.pt'):
                    continue
                    
                data = torch.load(os.path.join(sub_path, filename), weights_only=False)
                in_channels = data.num_node_features
                out_channels = int(data.y.max().item() + 1)
                
                if model_name == "XGBoost":
                    model = model_class(in_channels, 64, out_channels)
                    acc = model.fit_predict(data)
                else:
                    data = data.to(device)
                    model = model_class(in_channels, 64, out_channels).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                    
                    # Training Loop
                    model.train()
                    for _ in range(50):
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index)
                        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        optimizer.step()
                        
                    # Evaluation
                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index)
                        pred = out.argmax(dim=1)
                        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                        acc = int(correct) / int(data.test_mask.sum().item())
                        
                print(f"  [{subset.upper()}] {filename:<15} {acc * 100:.2f}%")

if __name__ == "__main__":
    run_benchmark()