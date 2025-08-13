import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def train(model, optimizer, train_loader, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            out = model(
                x_nodes=data.x,
                edge_index=data.edge_index,
                batch=data.batch,
                topo_map=data.topo_map.repeat(data.num_graphs, 1, 1, 1)
            ).view(-1)

            target = data.y.view(-1)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}")

    return avg_loss

def test(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            out = model(
                x_nodes=data.x,
                edge_index=data.edge_index,
                batch=data.batch,
                topo_map=data.topo_map.repeat(data.num_graphs, 1, 1, 1)
            ).view(-1)

            target = data.y.view(-1)
            loss = F.mse_loss(out, target)
            total_loss += loss.item()

            # Debug dimensioni per il primo batch
            if batch_idx == 0:
                print(f"[TEST] data.x shape: {data.x.shape}")
                print(f"[TEST] data.y shape: {data.y.shape}")
                print(f"[TEST] model output shape: {out.shape}")
                print(f"[TEST] target shape: {target.shape}")

    avg_loss = total_loss / len(loader)
    print(f"Test Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate(model, loader, device, log_scale=True):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(
                x_nodes=data.x,
                edge_index=data.edge_index,
                batch=data.batch,
                topo_map=data.topo_map.repeat(data.num_graphs, 1, 1, 1)
            ).view(-1)

            target = data.y.view(-1)
            y_true.append(target.cpu())
            y_pred.append(out.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    print(f"Valutazione - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")

    # Scatter plot predizioni vs target
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Target")
    plt.ylabel("Predizione")
    plt.title("Predizione vs Target")
    if log_scale:
        # Evita valori zero per log
        eps = 1e-6
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(left=max(y_true.min(), eps))
        plt.ylim(bottom=max(y_pred.min(), eps))
    plt.grid(True)
    plt.show()

    return mse, mae, rmse, r2