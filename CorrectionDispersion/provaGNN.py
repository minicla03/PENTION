import torch
import numpy as np
import pandas as pd
import os
import json
from MCxM import MCxM_GNN, MCxC_GNN_Sensor
from torch_geometric.loader import DataLoader

from graph_gen import create_graph, plot_graph
from model_train_function import train, evaluate
from plot_utils_gnn import plot_comparison_maps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "roma_italy_bbox.npy")
METADATA_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "roma_italy_metadata_bbox.json")
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-08-08_processed.csv")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
PROCESSED_GRAPHS_PATH = os.path.join(SCRIPT_DIR, "processed_graphs_sensor.pt")

if __name__ == "__main__":

    binary_map = np.load(BINARY_MAP_PATH)
    with open(METADATA_MAP_PATH, 'r') as f:
        metadata = json.load(f)
    datset_copy = pd.read_csv(DATASET_PATH)

    n_sensors=10

    #datset_copy['contratio_series'] = datset_copy['contratio_series'].map(lambda s: np.array(list(map(float, s.split(',')))))

    if os.path.exists(PROCESSED_GRAPHS_PATH):
        graph_list = torch.load(PROCESSED_GRAPHS_PATH, map_location='cpu', weights_only=False)
        print(graph_list[0])
        print(f"Loaded processed graphs from {PROCESSED_GRAPHS_PATH}")
    else:
        graph_list= create_graph(datset_copy, binary_map, metadata, n_sensors, sensor=True)
        torch.save(graph_list, PROCESSED_GRAPHS_PATH)
        print(f"Processed graphs saved to {PROCESSED_GRAPHS_PATH}")

    
    plot_graph(graph_list[0], binary_map, f"graph_example.png")

    #  Training setup 
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    split_idx = int(0.8 * len(graph_list))
    train_dataset = graph_list[:split_idx]
    test_dataset = graph_list[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    node_feature_dim = len(graph_list[0].x[0])
    mask = torch.tensor(binary_map, dtype=torch.float32)
    model = MCxC_GNN_Sensor(node_feature_dim=node_feature_dim, mask=mask, m=binary_map.shape[0])
    epoch=20

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili: {total_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- Main training loop ---
    loss_train = train(model,optimizer, train_loader, device, epochs=epoch)
    print(f"Epoch {epoch}/20 - Train Loss: {loss_train:.6f}")

    """val_loss, val_mae, val_rmse, val_r2 = """
    """ print(f"Final Evaluation — Train MSE: {loss_train:.6f} | Val MSE: {val_loss:.6f} | "
        f"MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f} | R²: {val_r2:.4f}")"""
    evaluate(model, test_loader, device, binary_map)

    # Visualizza mappa errore spaziale per un esempio del test set
    sample_data = test_dataset[0].to(device)
    model.eval()
    with torch.no_grad():
        pred = model(sample_data)
    H, W = binary_map.shape
    #plot_comparison_maps(sample_data, pred, H, W)