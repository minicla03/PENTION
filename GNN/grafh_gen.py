import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.loader import DataLoader

# === Percorsi ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "benevento_italy_full_map.npy")
METADATA_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "benevento_italy_metadata.npy")
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_0408_v4_processed.csv")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
PROCESSED_GRAPHS_PATH = os.path.join(SCRIPT_DIR, "processed_graphs.pt")

# === Funzione per creare un grafo ===
def create_graph_from_simulation(row, binary_map, t_idx=300):
    H, W = binary_map.shape
    num_nodes = H * W

    is_free = binary_map.flatten().astype(np.float32)

    # Coordinate griglia e distanza da sorgente
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dists = np.sqrt((xx - row['source_x'])**2 + (yy - row['source_y'])**2).flatten().astype(np.float32)

    # Carica mappa concentrazione
    CONC_PATH = os.path.join(REAL_CONC_PATH, row['real_concentration'])
    conc_array = np.array(np.load(CONC_PATH))
    conc_gauss = conc_array[:, :, t_idx].flatten().astype(np.float32)

    # Feature globali replicate
    wind_cos = np.full(num_nodes, row['wind_dir_cos'], dtype=np.float32)
    wind_sin = np.full(num_nodes, row['wind_dir_sin'], dtype=np.float32)
    wind_speed = np.full(num_nodes, row['wind_speed'], dtype=np.float32)
    emission_rate = np.full(num_nodes, row['emission_rate'], dtype=np.float32)
    source_h = np.full(num_nodes, row['source_h'], dtype=np.float32)

    aerosol_id = hash(row['aerosol_type']) % 10
    stability_id = hash(row['stability_value']) % 10
    aerosol_type = np.full(num_nodes, aerosol_id, dtype=np.float32)
    stability_value = np.full(num_nodes, stability_id, dtype=np.float32)

    node_features = np.stack([
        is_free, dists, conc_gauss,
        wind_cos, wind_sin, wind_speed,
        emission_rate, source_h, aerosol_type, stability_value
    ], axis=1)

    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor(conc_gauss, dtype=torch.float32).unsqueeze(1)
    pos = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)
    edge_index, _ = pyg_utils.grid(H, W)

    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
    data.mask = torch.tensor(is_free, dtype=torch.float32)  
    return data

def compute_saliency(model, data, device):
    model.eval()
    data = data.to(device)
    data.x.requires_grad_()

    out = model(data)
    # Considera somma output per semplificare gradiente scalare
    score = out.sum()
    model.zero_grad()
    score.backward()

    saliency = data.x.grad.abs().cpu().numpy()  # [num_nodes, num_features]
    return saliency

def plot_saliency_on_grid(saliency, feature_idx, H, W, title="Saliency Map"):
    # saliency: numpy [num_nodes, num_features]
    saliency_feat = saliency[:, feature_idx].reshape(H, W)
    plt.figure(figsize=(8,6))
    im = plt.imshow(saliency_feat, cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Importanza Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# === Modello con BatchNorm e Dropout + MaskLayer e Pooling globale ===
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

        # Predizione nodo per nodo
        out_node = self.output_layer(x)

        # rappresentazione globale aggregata
        graph_repr = self.global_pool(x, batch)

        return out_node 
    
# === Funzione per visualizzare mappa errore spaziale 2D ===
def plot_error_map(data, prediction, H, W, title="Errore spaziale (predizione - target)"):
    # data: oggetto Data PyG con pos
    # prediction: tensor [num_nodes,1]
    error = prediction.detach().cpu().numpy().flatten() - data.y.detach().cpu().numpy().flatten()
    error_map = error.reshape(H, W)

    plt.figure(figsize=(8,6))
    im = plt.imshow(error_map, cmap='bwr', interpolation='nearest')
    plt.colorbar(im, label='Errore concentrazione')
    plt.title(title)
    plt.xlabel('X griglia')
    plt.ylabel('Y griglia')
    plt.gca().invert_yaxis()
    plt.show()

# === Main ===
if __name__ == "__main__":
    binary_map = np.load(BINARY_MAP_PATH)
    metadata = np.load(METADATA_MAP_PATH, allow_pickle=True).item()
    datset_copy = pd.read_csv(DATASET_PATH)

    if os.path.exists(PROCESSED_GRAPHS_PATH):
        graph_list = torch.load(PROCESSED_GRAPHS_PATH, map_location='cpu', weights_only=False)
    else:
        graph_list = []
        for idx, row in tqdm(datset_copy.iterrows(), total=len(datset_copy), desc="Processing simulations"):
            graph = create_graph_from_simulation(row, binary_map)
            graph_list.append(graph)
            if idx == 10:  
                break
        torch.save(graph_list, PROCESSED_GRAPHS_PATH)

    # === Training setup ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    split_idx = int(0.8 * len(graph_list))
    train_dataset = graph_list[:split_idx]
    test_dataset = graph_list[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    model = MCxM_GNN(in_channels=train_dataset[0].x.shape[1], hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
        return total_loss / len(train_loader.dataset) / data.num_nodes

    def test(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                loss = F.mse_loss(out, data.y)
                total_loss += loss.item() * data.num_nodes
        return total_loss / len(loader.dataset) / data.num_nodes # type: ignore

    for epoch in range(1, 21):
        loss_train = train()
        print(f"Epoch {epoch}/20 - Train Loss: {loss_train:.6f}")

    def evaluate(model, loader, device):
        model.eval()
        total_loss = 0
        total_mae = 0
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                loss = F.mse_loss(out, data.y)
                total_loss += loss.item() * data.num_nodes
                mae = F.l1_loss(out, data.y)
                total_mae += mae.item() * data.num_nodes
                y_true_list.append(data.y.cpu().numpy())
                y_pred_list.append(out.cpu().numpy())
        y_true = np.concatenate(y_true_list).flatten()
        y_pred = np.concatenate(y_pred_list).flatten()

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        n = len(y_true)
        avg_loss = total_loss / n
        avg_mae = total_mae / n
        return avg_loss, avg_mae, rmse, r2

    val_loss, val_mae, val_rmse, val_r2 = evaluate(model, test_loader, device)
    print(f"Final Evaluation — Train MSE: {loss_train:.6f} | Val MSE: {val_loss:.6f} | " # type: ignore
          f"MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f} | R²: {val_r2:.4f}")

    # === Visualizza mappa errore spaziale per un esempio del test set ===
    sample_data = test_dataset[0].to(device)
    model.eval()
    with torch.no_grad():
        pred = model(sample_data)
    H, W = binary_map.shape
    plot_error_map(sample_data, pred, H, W, title="Errore spaziale esempio test")

    sample_data = test_dataset[0]
    saliency = compute_saliency(model, sample_data, device)

    # Per esempio, visualizza la saliency sulla feature 'dists' che è index 1 nel vettore features
    plot_saliency_on_grid(saliency, feature_idx=1, H=binary_map.shape[0], W=binary_map.shape[1], title="Saliency sulla distanza dalla sorgente")
