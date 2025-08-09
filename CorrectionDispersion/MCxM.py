# GNN/MCxM.py
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import BatchNorm

class MaskLayer(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        mask= torch.tensor(mask, dtype=torch.float32)
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        return x * self.mask

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

        return out_graph.view() 

class MCxM_CNN(torch.nn.Module):
    def __init__(self, mask, m=500):
        super().__init__()

        input_size = m*m+2
        hidden_size = 512

        self.mask_layer = MaskLayer(mask)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),       # 10000 → 512
            nn.BatchNorm1d(hidden_size),              # BN prima di ReLU aiuta a mantenere la media e la varianza dei dati stabili durante l’addestramento.
            nn.ReLU(),                                  #Metterla prima e dopo la ReLU serve a mantenere i dati ben distribuiti, riducendo problemi di saturazione o distribuzioni sbilanciate.
            nn.BatchNorm1d(hidden_size),              # BN dopo ReLU

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )

        # DECODER: riportiamo a m²
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),       # 512 → 10000
            nn.Sigmoid()  # normalizza output tra 0 e 1, utile per mappe binarie
        )

        self.m = m  # salviamo m per il reshape
    
    def forward(self, mc, wind_speed, wind_dir):
        B = mc.size(0)
    
        # 1. Flatten la mappa
        x = mc.view(B, -1)  # shape: (B, 10000)

        # 2. Concateniamo vento e direzione
        wind_features = torch.cat([wind_speed, wind_dir], dim=1)  # shape: (B, 2)

        # 3. Concatenazione finale
        x = torch.cat([x, wind_features], dim=1)  # shape: (B, 10002)

        # 4. Passiamo nell’encoder-decoder
        x = self.encoder(x)
        x = self.decoder(x)

        # 5. Reshape finale (solo la parte mappa)
        x = x.view(B, self.m, self.m)

        return x
