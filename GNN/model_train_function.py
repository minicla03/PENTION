import torch.nn.functional as F
import torch
import numpy as np

def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data).view(-1)   # output [batch_size]
        target = data.y.view(-1)     # target [batch_size]

        # Debug dimensioni
        if batch_idx == 0:
            print(f"[TRAIN] data.x shape: {data.x.shape}")
            print(f"[TRAIN] data.y shape: {data.y.shape}")
            print(f"[TRAIN] model output shape: {out.shape}")
            print(f"[TRAIN] target shape: {target.shape}")

        # Backward pass
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            out = model(data).view(-1)
            target = data.y.view(-1)

            # Debug dimensioni
            if batch_idx == 0:
                print(f"[TEST] data.x shape: {data.x.shape}")
                print(f"[TEST] data.y shape: {data.y.shape}")
                print(f"[TEST] model output shape: {out.shape}")
                print(f"[TEST] target shape: {target.shape}")

            loss = F.mse_loss(out, target)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, binary_map):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()      # [batch_size]
            target = data.y.view(-1)         # [batch_size]

            y_true.append(target.cpu())
            y_pred.append(out.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    print(f"Valutazione - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")

    # Visualizza predizioni vs target (scatter plot)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Target")
    plt.ylabel("Predizione")
    plt.title("Predizione vs Target")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()