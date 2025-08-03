import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SourceLocatorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x=F.relu(self.conv1(x, edge_index))
        x=F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  
        return self.linear(x)