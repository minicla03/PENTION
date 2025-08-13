# GNN/MCxM.py
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import BatchNorm
#from torch_scatter import scatter_add

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
    

class MCxC_GNN_Sensor(nn.Module):
    def __init__(self, node_feature_dim, hidden_size=128, topo_embed_size=64, dropout_p=0.3, mask=None, m=50):
        """
        node_feature_dim: dimensione delle feature dei nodi (es. 12 se includi building_density, mean_height ecc.)
        topo_embed_size: dimensione embedding globale della topografia
        m: dimensione della griglia finale
        """
        super().__init__()
        self.m = m
        self.node_feature_dim = node_feature_dim

        # Encoder topografia: CNN → embedding globale
        self.topo_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, topo_embed_size, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)  # (B, topo_embed_size, 1, 1)
        )

        # GNN layers
        self.conv1 = GCNConv(node_feature_dim + topo_embed_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        # Decoder nodo → stima dispersione
        self.node_decoder = nn.Linear(hidden_size, 1)

        # Maschera della città sulla griglia finale
        self.mask_layer = MaskLayer(mask)

    def forward(self, x_nodes, edge_index, batch, topo_map):
        """
        x_nodes: (num_nodes, node_feature_dim)
        edge_index: edge_index concatenato per il batch
        batch: tensor con batch assignment dei nodi
        topo_map: (B, 1, m, m) mappa binaria
        """
        # --- Embedding globale topografia ---
        topo_feat = self.topo_encoder(topo_map).view(topo_map.size(0), -1)  # (B, topo_embed_size)
        topo_feat_nodes = topo_feat[batch]  # broadcast a tutti i nodi
        x = torch.cat([x_nodes, topo_feat_nodes], dim=1)  # (num_nodes, node_feature_dim + topo_embed_size)

        # --- Passaggio nella GNN ---
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # --- Decoder nodo → stima dispersione ---
        node_out = self.node_decoder(x)  # (num_nodes, 1)

        # --- Interpolazione su griglia m×m usando scatter_add ---
        B = topo_map.size(0)
        grid = torch.zeros(B, self.m, self.m, device=x.device)

        # Coord. normalizzate [0,1] → pixel [0, m-1]
        coords = x_nodes[:, 1:3]  # assume x_coord, y_coord normalizzate
        xi = (coords[:, 0] * (self.m - 1)).long().clamp(0, self.m - 1)
        yi = (coords[:, 1] * (self.m - 1)).long().clamp(0, self.m - 1)

        # Scatter add per evitare sovrascritture multiple nello stesso pixel
        scatter_add(node_out[:, 0], batch * self.m * self.m + yi * self.m + xi, out=grid.view(-1))

        # --- Applica maschera della città ---
        grid = self.mask_layer(grid)

        return grid

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.verbose = True

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"[EarlyStopping] Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement in validation loss for {self.counter} epochs")
            if self.counter >= self.patience:
                self.early_stop = True

class MCxM_CNN(torch.nn.Module):
    def __init__(self, mask, m=500, dropout_p=0.3, n_channel=2, n_global_features=3):
        super().__init__()

        input_size = m*m+2
        hidden_size = 512

        self.mask_layer = MaskLayer(mask)
        self.dropout = nn.Dropout(p=dropout_p)


        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),       # m*m+2 → 512
            nn.BatchNorm1d(hidden_size),              # BN prima di ReLU aiuta a mantenere la media e la varianza dei dati stabili durante l’addestramento.
            nn.ReLU(),                                  #Metterla prima e dopo la ReLU serve a mantenere i dati ben distribuiti, riducendo problemi di saturazione o distribuzioni sbilanciate.
            self.dropout,                          # Dropout per regolarizzazione
            nn.BatchNorm1d(hidden_size),              # BN dopo ReLU

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),      # 512 → 512
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_size),
        )

        # DECODER: riportiamo a m²
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, m*m),       # 512 → m*m
            nn.Sigmoid()  # normalizza output tra 0 e 1, utile per mappe binarie
        )

        self.m = m  # salviamo m per il reshape
    
    def forward(self, mc, wind_speed, wind_dir):
        B = mc.size(0)

        #print(f"[DEBUG CNN] Input mc shape: {mc.shape}")
        # 1. Flatten la mappa
        x = mc.view(B, -1)  # shape: (B, m*m)

        # 2. Concateniamo vento e direzione
        wind_features = torch.cat([wind_speed, wind_dir], dim=1)  # shape: (B, 2)
        #print(f"[DEBUG CNN] Wind features shape: {wind_features.shape}")


        # 3. Concatenazione finale
        x = torch.cat([x, wind_features], dim=1)  # shape: (B, m*m+2)
        #print(f"[DEBUG CNN] After concat with wind, x shape: {x.shape}")


        # 4. Passiamo nell’encoder-decoder
        x = self.encoder(x)
        #print(f"[DEBUG CNN] After decoder, x shape: {x.shape}")

        x = self.decoder(x)
        #print(f"[DEBUG CNN] After reshape, x shape: {x.shape}")

        # 5. Reshape finale (solo la parte mappa)
        x = x.view(B, self.m, self.m)
        #print(f"[DEBUG CNN] After mask layer, x shape: {x.shape}")


        # Applico maschera anche in output
        x = self.mask_layer(x)
        #print(f"[DEBUG CNN] After mask layer, x shape: {x.shape}")

        return x
    
class MCxM_CNN_Hybrid(torch.nn.Module):
    def __init__(self, mask, m=500, dropout_p=0.3, n_local_channels=2, n_global_features=6):
        super().__init__()

        self.m = m
        self.mask_layer = MaskLayer(mask)
        self.dropout = nn.Dropout(p=dropout_p)

        # Dimensione input: m*m*n_local_channels + n_global_features + 2 (vento)
        input_size = m * m * n_local_channels + n_global_features + 2
        hidden_size = 512

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_size),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.BatchNorm1d(hidden_size),
        )

        # Decoder: output m*m (la mappa corretta)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, m * m),
            nn.Sigmoid()
        )

    def forward(self, local_maps, wind_speed, wind_dir, global_features):
        """
        local_maps: tensor [B, n_channels, m, m]
        wind_speed, wind_dir: [B,1]
        global_features: [B, n_global_features]
        """
        B = local_maps.size(0)

        # 1. Flatten le mappe locali
        x_local = local_maps.view(B, -1)  # [B, m*m*n_local_channels]

        # 2. Concatenazione vento e parametri globali
        x = torch.cat([x_local, wind_speed, wind_dir, global_features], dim=1)  # [B, m*m*n_channels + n_global_features + 2]

        # 3. Passaggio encoder-decoder
        x = self.encoder(x)
        x = self.decoder(x)

        # 4. Reshape in m*m
        x = x.view(B, self.m, self.m)

        # 5. Applica maschera finale
        x = self.mask_layer(x)

        return x

