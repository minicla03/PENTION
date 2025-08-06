# GNN/MCxM.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import BatchNorm

class MaskLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, mask):
        return x * mask.unsqueeze(1)

class MCxM_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_corrections=3, dropout_p=0.3):
        super().__init__()
        self.num_corrections = num_corrections
        self.mask_layer = MaskLayer()
        self.gcn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout_p = dropout_p

        for i in range(num_corrections):
            input_dim = in_channels if i == 0 else hidden_channels
            self.gcn_layers.append(GCNConv(input_dim, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        self.output_layer = torch.nn.Linear(hidden_channels, 1)
        self.global_pool = global_mean_pool

    def forward(self, data):
        x, edge_index, mask = data.x, data.edge_index, data.mask
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i in range(self.num_corrections):
            x = self.mask_layer(x, mask)
            x = self.gcn_layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.mask_layer(x, mask)
        graph_repr = self.global_pool(x, batch)  # [num_graphs, hidden_channels]

        out_graph = self.output_layer(graph_repr)  # [num_graphs, 1]

        return out_graph.squeeze()  