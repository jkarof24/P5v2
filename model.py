import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv

try:
    import xgboost as XGB
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier


 
# 1. XGBoost Baseline
 
class XGBoostModel:
    def __init__(self, in_channels, hidden_channels, out_channels):
        self.out_channels = out_channels

    def fit_predict(self, data):
        X_train = data.x[data.train_mask].cpu().numpy()
        y_train = data.y[data.train_mask].cpu().numpy()
        X_test = data.x[data.test_mask].cpu().numpy()
        y_test = data.y[data.test_mask].cpu().numpy()

        if HAS_XGB:
            model = XGB.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
        else:
            model = GradientBoostingClassifier(n_estimators=50, random_state=42)

        unique_classes = np.unique(y_train)
        class_map = {val: idx for idx, val in enumerate(unique_classes)}
        y_train_mapped = np.array([class_map[v] for v in y_train])

        model.fit(X_train, y_train_mapped)
        preds_mapped = model.predict(X_test)

        inv_map = {idx: val for val, idx in class_map.items()}
        preds = np.array([inv_map.get(p, unique_classes[0]) for p in preds_mapped])

        correct = (preds == y_test).sum()
        return correct / len(y_test)


 
# 2. Standard GCN
 
class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, out_c)

    def forward(self, x, edge_index):
        x = F.dropout(self.conv1(x, edge_index).relu(), p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)


 
# 3. GraphSAGE
 
class GraphSAGE(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hid_c)
        self.conv2 = SAGEConv(hid_c, out_c)

    def forward(self, x, edge_index):
        x = F.dropout(self.conv1(x, edge_index).relu(), p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)


 
# 4. GCNII (Deep GCN - Fixed layer 1..N)
 
class GCNII(nn.Module):
    def __init__(self, in_c, hid_c, out_c, layers=8):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(in_c, hid_c), nn.Linear(hid_c, out_c)])
        # FIX: layer must be 1-indexed (i + 1) to avoid ZeroDivisionError
        self.convs = nn.ModuleList([GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=i + 1, cached=True) for i in range(layers)])

    def forward(self, x, edge_index):
        x0 = F.dropout(x, p=0.5, training=self.training).relu()
        x0 = self.lins[0](x0)
        x = x0
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, x0, edge_index).relu()
        return F.log_softmax(self.lins[1](F.dropout(x, p=0.5, training=self.training)), dim=1)


 
# 5. PolyFormer (Spectral Attention)
 
class PolyFormer(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.lin1 = nn.Linear(in_c, hid_c)
        self.conv1 = GCNConv(hid_c, hid_c)
        self.att = nn.MultiheadAttention(embed_dim=hid_c, num_heads=4, batch_first=True)
        self.lin2 = nn.Linear(hid_c, out_c)

    def forward(self, x, edge_index):
        h = self.conv1(self.lin1(x).relu(), edge_index).relu().unsqueeze(1)
        h, _ = self.att(h, h, h)
        return F.log_softmax(self.lin2(h.squeeze(1)), dim=1)


 
# 6. SwapGT (Token-Swapping Graph Transformer)

class SwapGTModel(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.lin_in = nn.Linear(in_c, hid_c)
        self.conv = GCNConv(hid_c, hid_c)
        self.transformer = nn.TransformerEncoderLayer(d_model=hid_c, nhead=4, batch_first=True)
        self.lin_out = nn.Linear(hid_c, out_c)

    def forward(self, x, edge_index):
        h = self.lin_in(x).relu()
        h_conv = self.conv(h, edge_index).relu()
        h_seq = h_conv.unsqueeze(1)
        h_trans = self.transformer(h_seq).squeeze(1)
        return F.log_softmax(self.lin_out(h_trans), dim=1)