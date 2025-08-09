import numpy as np
import matplotlib.pyplot as plt

def plot_comparison_maps(data, pred, H, W):
    pred_np = pred.detach().cpu().numpy()
    target_np = data.y.detach().cpu().numpy()

    # Assicurati che pred e target abbiano la stessa dimensione
    assert pred_np.size == target_np.size == H * W, "Dimensione predizione e target devono essere H*W"

    pred_map = pred_np.reshape(H, W)
    target_map = target_np.reshape(H, W)
    error_map = pred_map - target_map

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axs[0].imshow(target_map, cmap='viridis')
    axs[0].set_title("Target")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred_map, cmap='viridis')
    axs[1].set_title("Predizione")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(error_map, cmap='bwr', vmin=-np.max(np.abs(error_map)), vmax=np.max(np.abs(error_map)))
    axs[2].set_title("Errore spaziale (Pred - Target)")
    plt.colorbar(im2, ax=axs[2])

    plt.show()