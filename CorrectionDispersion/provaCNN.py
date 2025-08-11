from MCxM import MCxM_CNN, EarlyStopping
#from pytorch_lightning.callbacks import EarlyStopping
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

def masked_loss(output, target, mask, lambda_penalty=10.0):

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=output.device)
    else:
        mask = mask.to(output.device)

    # Loss sulle zone libere (mask=1)
    mse_loss = torch.mean(((output - target) ** 2) * mask)

    # Penalizzazione presenza negli edifici (mask=0)
    building_penalty = torch.mean((output * (1 - mask)) ** 2)

    # Loss totale con bilanciamento
    total_loss = mse_loss + lambda_penalty * building_penalty

    return total_loss

class CNNDataset(Dataset):
    def __init__(self, concentration_maps, wind_dir, wind_speed, m=500, path_saved="./CorrectionDispersion/CNNdataset"):
        super().__init__(root=path_saved)
        self.concentration_maps = concentration_maps  # lista o array di (m, m)
        self.wind_dir = wind_dir                      # lista o array di scalari
        self.wind_speed = wind_speed
        self.m = m

    def __len__(self):
        return len(self.concentration_maps)

    def __getitem__(self, idx):
        mc = torch.tensor(self.concentration_maps[idx], dtype=torch.float32).view(self.m, self.m)
        wind_speed = torch.tensor([self.wind_speed[idx]], dtype=torch.float32)  # shape (1,)
        wind_dir   = torch.tensor([self.wind_dir[idx]], dtype=torch.float32)    # shape (1,)
        return mc, wind_speed, wind_dir


def r2_score(y_true, y_pred, eps=1e-10):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2) + eps
        print(f"ss_res: {ss_res.item()}, ss_tot: {ss_tot.item()}")
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
            
            mc_orig = denormalize_tensor(mc, vmin, vmax)
            output_orig = denormalize_tensor(output, vmin, vmax)

            r2 = r2_score(mc_orig, output_orig)

            total_loss += loss.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    avg_r2 = total_r2 / n_batches

    print(f"Validation - Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}, R2: {avg_r2:.6f}")
    return avg_loss, avg_mae, avg_r2

def convert_string_to_array(s):
    if isinstance(s, str):
        s = s.replace('[', '').replace(']', '')
        return np.fromstring(s, sep=',')
    return s

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from windrose import WindroseAxes


def plot_plan_view(C1, x, y, title, wind_dir=None, wind_speed=None, puff_list=None, stability_class=1, n_show=10):
    fig, ax_main = plt.subplots(figsize=(8, 6))

    # Integra la concentrazione nel tempo lungo l'asse 2 (T)
    #data = np.trapz(C1, axis=2) * 1e6  # µg/m³ #type:ignore

    if isinstance(C1, torch.Tensor):
        C1 = C1.detach().cpu().numpy()
    
    data=C1
    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 95)

    # Plot della concentrazione integrata
    pcm = ax_main.pcolor(x, y, data, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax_main, label=r'$\mu g \cdot m^{-3}$')
    ax_main.set_xlabel('x (m)')
    ax_main.set_ylabel('y (m)')
    ax_main.set_title(title)
    ax_main.axis('equal')

    if wind_dir is not None and wind_speed is not None:
        inset_pos = [0.65, 0.65, 0.3, 0.3]  # left, bottom, width, height in figure coords
        ax_inset = WindroseAxes(fig, inset_pos)
        fig.add_axes(ax_inset)

        # Plot rosa dei venti con direzioni e velocità
        wind_dir = np.array(wind_dir) % 360
        wind_speed = np.full_like(wind_dir, fill_value=wind_speed, dtype=float)
        ax_inset.bar(wind_dir, wind_speed, normed=True, opening=0.8, edgecolor='white')
        ax_inset.set_legend(loc='lower right', title='Wind speed (m/s)')
        ax_inset.set_title("Rosa dei venti")

    # Plot puff sopra la plan view
    if puff_list is not None and len(puff_list) > 0:
        # Parametri σ_y empirici per classi A-F (Pasquill-Gifford)
        a_vals = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
        b_vals = [0.90, 0.88, 0.86, 0.83, 0.80, 0.78]
        a = a_vals[stability_class - 1]
        b = b_vals[stability_class - 1]

        for i, puff in enumerate(puff_list):
            # if i % n_show != 0:
            #     continue  # salta puff intermedi

            distance = np.sqrt(puff.x**2 + puff.y**2)
            sigma_y = a * (distance + 1)**b  # evita 0^b

            circle = Circle((puff.x, puff.y), 2 * sigma_y, color='white', fill=False, lw=1.5)
            ax_main.add_patch(circle)
            ax_main.plot(puff.x, puff.y, 'wo', markersize=3)

        ax_main.legend(["Puff center (2σ)"], loc='lower right')

    plt.tight_layout()
    plt.show()

def normalize_tensor(tensor, vmin, vmax):
    return (tensor - vmin) / (vmax - vmin)

def denormalize_tensor(tensor, vmin, vmax):
    return tensor * (vmax - vmin) + vmin

if __name__ == "__main__":
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")
    REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
    CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-08-08_processed.csv")

    binary_map = np.load(BINARY_MAP_PATH)
    csv_df = pd.read_csv(CSV_PATH)
    csv_df_reduced = csv_df.groupby('simulation_id').first().reset_index()
    csv_df_reduced = csv_df_reduced[['wind_dir', 'wind_speed']]
    m = 500

    csv_df_reduced['wind_dir'] = csv_df_reduced['wind_dir'].apply(convert_string_to_array) #type:ignore

    concentration_maps = [] #ground true
    wind_dirs = []
    wind_speeds = []
    for file in tqdm(os.listdir(REAL_CONC_PATH), desc="Loading concentration maps"):
        conc_map = np.load(os.path.join(REAL_CONC_PATH, file))
        conc_map_mean = np.mean(conc_map, axis=2)  # Assuming conc_map is of shape (m, m, n)
        i = int(file.split('_')[1])  # file name format is 'sim_i_conc_real_...'
        wind_dir, wind_speed = csv_df_reduced.iloc[i]  # correct order: dataframe columns are wind_dir, wind_speed
        
        concentration_maps.append(conc_map_mean)

        wind_speed_mean= np.mean(wind_speed)  # Assuming wind_dir is an array
        
        wind_dir = np.array(wind_dir) % 360  # normalizza
        # Media direzione (media circolare)
        rad = np.deg2rad(wind_dir)
        x = np.cos(rad)
        y = np.sin(rad)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        mean_dir_rad = np.arctan2(y_mean, x_mean)
        mean_dir_deg = np.rad2deg(mean_dir_rad) % 360  # mantieni in [0,360)

        wind_dirs.append(mean_dir_deg)
        wind_speeds.append(wind_speed_mean)
    dati = list(zip(concentration_maps, wind_dirs, wind_speeds))
    
    """print("[INFO] Plot concentration maps and wind parameters.")
    binary_map= binary_map[:,:, np.newaxis]
    plot_plan_view((dati[0][0])*binary_map, np.arange(m), np.arange(m), "Example Concentration Map", 
                   wind_dir=(dati[0][1]).astype(float), wind_speed=(dati[0][2]).astype(float), puff_list=None, stability_class=1)
    """

    all_values = np.concatenate([cm.flatten() for cm in concentration_maps])
    vmin, vmax = np.min(all_values), np.max(all_values)
    print(f"Global min: {vmin}, max: {vmax}")

    concentration_maps = [(cm - vmin) / (vmax - vmin) for cm in concentration_maps]
        
    print("[INFO] Initializing CNNDataset.")
    dataset = CNNDataset(concentration_maps, wind_dirs, wind_speeds, m=m)

    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) #datafusion
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) 

    print("Statistiche sul dataset:")
    for i in range(3):
        mc_sample, ws, wd = dataset[i]
        print(f"Sample {i} - mc min: {mc_sample.min().item()}, max: {mc_sample.max().item()}, mean: {mc_sample.mean().item()}")


    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[INFO] Using device: {device}")

    epochs=10
    model = MCxM_CNN(binary_map, m=m).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    decay_rate = 0.95
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    early_stopper = EarlyStopping(patience=5)

    train_losses  = []
    val_losses = []

    def plot_losses():
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.ylabel('Loss')
        plt.title('Andamento della Loss durante il Training')
        plt.legend()
        plt.grid(True)
        plt.show()

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train()
        running_loss = 0.0
        for batch_idx, (mc, wind_speed, wind_dir) in enumerate(train_loader):
            mc = mc.to(device)
            wind_speed = wind_speed.to(device)
            wind_dir = wind_dir.to(device)

            optimizer.zero_grad()
            output = model(mc, wind_speed, wind_dir)

            print(f"[DEBUG Training] Epoch {epoch+1}, Batch {batch_idx+1}")
            print(f"  Input mc shape: {mc.shape}")
            print(f"  Output shape: {output.shape}")

            loss = masked_loss(output, mc, binary_map)
            #plot_plan_view(output, np.arange(m), np.arange(m), "output")
            print(f"  Loss: {loss.item():.6f}")
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
        scheduler.step()
       
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[INFO] Epoch {epoch+1} average training loss: {avg_train_loss:.6f}")

        val_loss, val_mae, val_r2  = validate(model, val_loader, loss_fn, torch.nn.L1Loss(), device)  

        val_losses.append(val_loss)

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopper.best_model_state)
            break 

    plot_losses()
    
