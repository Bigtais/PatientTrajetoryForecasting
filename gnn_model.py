import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# ==== Define R-GCN encoder ====
class RGCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_rels)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_rels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        return self.conv2(x, edge_index, edge_type)
