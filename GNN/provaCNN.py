from MCxM import MCxM_CNN
import torch
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gaussianPuff')))
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pandas as pd
from plot_utils import plot_plan_view
from tqdm import tqdm

class CNNDataset(Dataset):
    def __init__(self, concentration_maps, wind_dir, wind_speed, m=500, path_saved="./GNN/CNNdataset"):
        super().__init__(root=path_saved)
        self.concentration_maps = concentration_maps  # lista o array di (m, m)
        self.wind_dir = wind_dir                      # lista o array di scalari
        self.wind_speed = wind_speed
        self.m = m

    def __len__(self):
        return len(self.concentration_maps)

    def __getitem__(self, idx):
        mc = torch.tensor(self.concentration_maps[idx], dtype=torch.float32).view(1, self.m, self.m)
        wind_speed = torch.tensor([[self.wind_speed[idx]]], dtype=torch.float32)
        wind_dir = torch.tensor([[self.wind_dir[idx]]], dtype=torch.float32)
        return mc, wind_speed, wind_dir


def r2_score(y_true, y_pred):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

def validate(model, val_loader, loss_fn, mae_fn, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_r2 = 0
    n_batches = 0

    with torch.no_grad():
        for mc, wind_speed, wind_dir in val_loader:
            mc = mc.to(device)
            wind_speed = wind_speed.to(device)
            wind_dir = wind_dir.to(device)

            output = model(mc, wind_speed, wind_dir)
            loss = loss_fn(output, mc)
            mae = mae_fn(output, mc)
            r2 = r2_score(mc, output)

            total_loss += loss.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    avg_r2 = total_r2 / n_batches

    print(f"Validation - Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}, R2: {avg_r2:.6f}")
    return avg_loss, avg_mae, avg_r2

if __name__ == "__main__":
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")
    REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
    CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-08-08.csv")

    binary_map = np.load(BINARY_MAP_PATH)
    csv_df = pd.read_csv(CSV_PATH)
    csv_df_reduced = csv_df.groupby('simulation_id').first().reset_index()
    csv_df_reduced = csv_df_reduced[['wind_speed', 'wind_dir']]
    m = 500

    dati = []
    for file in tqdm(os.listdir(REAL_CONC_PATH), desc="Loading concentration maps"):
        conc_map = np.load(os.path.join(REAL_CONC_PATH, file))
        i = int(file.split('_')[1])  # Assuming file name format is 'sim_i_conc_real_...'
        wind_speed, wind_dir= csv_df_reduced.iloc[i]
        dati.append((conc_map, wind_speed, wind_dir))
    
    print("[INFO] Plot concentration maps and wind parameters.")
    binary_map= binary_map[:,:, np.newaxis]
    plot_plan_view((dati[0][0])*binary_map, np.arange(m), np.arange(m), "Example Concentration Map", 
                   wind_dir=dati[0][2], wind_speed=dati[0][1], puff_list=None, stability_class=1)

    print("[INFO] Initializing CNNDataset.")
    dataset = CNNDataset(*dati, m=500)

    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) #datafusion
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) 

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[INFO] Using device: {device}")

    epochs=10
    model = MCxM_CNN(binary_map, m=m).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        for mc, wind_speed, wind_dir in train_loader:
            mc = mc.to(device)
            wind_speed = wind_speed.to(device)
            wind_dir = wind_dir.to(device)

            optimizer.zero_grad()
            output = model(mc, wind_speed, wind_dir)
            loss = loss_fn(output, mc)
            loss.backward()
            optimizer.step()

        validate(model, val_loader, loss_fn, torch.nn.L1Loss(), device)   
    
