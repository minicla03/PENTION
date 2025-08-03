import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class DispersionCNN(nn.Module):
    """
    CNN per predire la dispersione di gas in ambiente urbano
    Input: mappa binaria + posizione sorgente
    Output: campo di concentrazione
    """
    
    def __init__(self, grid_size=300):
        super(DispersionCNN, self).__init__()
        self.grid_size = grid_size
        
        # Encoder: estrae features dalla mappa
        self.encoder = nn.Sequential(
            # Prima convoluzione: mappa binaria + source mask
            nn.Conv2d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Secondo livello
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Terzo livello
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # Decoder: genera campo di concentrazione
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # Output finale: concentrazione
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Concentrazione normalizzata [0,1]
        )
    
    def forward(self, binary_map, source_positions):
        """
        Args:
            binary_map: (batch, 1, H, W) - mappa binaria edifici
            source_positions: (batch, 2) - coordinate [x, y] sorgenti
        Returns:
            concentration: (batch, 1, H, W) - campo concentrazione
        """
        batch_size = binary_map.shape[0]
        
        # Crea source mask
        source_mask = torch.zeros_like(binary_map)
        for i, (x, y) in enumerate(source_positions):
            source_mask[i, 0, int(y), int(x)] = 1.0
        
        # Combina mappa + source
        input_tensor = torch.cat([binary_map, source_mask], dim=1)
        
        # Forward pass
        features = self.encoder(input_tensor)
        concentration = self.decoder(features)
        
        return concentration


class GaussianDispersionSimulator:
    """
    Simulatore semplificato di dispersione gaussiana con ostacoli
    """
    
    def __init__(self, grid_size=300):
        self.grid_size = grid_size
    
    def simulate(self, binary_map, source_x, source_y, 
                 sigma=20, wind_x=0, wind_y=0):
        """
        Simula dispersione gaussiana con effetto edifici
        
        Args:
            binary_map: mappa binaria (1=libero, 0=edificio)
            source_x, source_y: posizione sorgente
            sigma: deviazione standard gaussiana
            wind_x, wind_y: vento (shift del centro)
        """
        # Crea griglia coordinate
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        
        # Centro con effetto vento
        center_x = source_x + wind_x
        center_y = source_y + wind_y
        
        # Dispersione gaussiana base
        gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Effetto edifici: riducono concentrazione
        # Gli edifici bloccano/riducono la dispersione
        building_effect = binary_map.astype(float)
        building_effect = gaussian_filter(building_effect, sigma=2)  # Smooth edges
        
        # Concentrazione finale
        concentration = gaussian * building_effect
        
        # Normalizza
        if concentration.max() > 0:
            concentration = concentration / concentration.max()
        
        return concentration


def generate_training_dataset(binary_map, n_samples=1000):
    """
    Genera dataset di training per la rete
    """
    simulator = GaussianDispersionSimulator(binary_map.shape[0])
    
    X_maps = []
    X_sources = []
    y_concentrations = []
    
    print(f"Generando {n_samples} campioni di training...")
    
    for i in range(n_samples):
        # Posizione casuale in spazio libero
        free_cells = np.where(binary_map == 1)
        idx = np.random.randint(0, len(free_cells[0]))
        source_x = free_cells[1][idx]  # x
        source_y = free_cells[0][idx]  # y
        
        # Parametri casuali
        sigma = np.random.uniform(10, 40)
        wind_x = np.random.uniform(-10, 10)
        wind_y = np.random.uniform(-10, 10)
        
        # Simula dispersione
        concentration = simulator.simulate(
            binary_map, source_x, source_y, sigma, wind_x, wind_y
        )
        
        X_maps.append(binary_map)
        X_sources.append([source_x, source_y])
        y_concentrations.append(concentration)
        
        if (i + 1) % 100 == 0:
            print(f"Completati {i + 1}/{n_samples} campioni")
    
    return (np.array(X_maps), np.array(X_sources), np.array(y_concentrations))


def train_model(model, X_maps, X_sources, y_concentrations, epochs=50):
    """
    Training loop semplificato
    """
    # Converte in tensori PyTorch
    X_maps_tensor = torch.FloatTensor(X_maps).unsqueeze(1)  # Add channel dim
    X_sources_tensor = torch.FloatTensor(X_sources)
    y_tensor = torch.FloatTensor(y_concentrations).unsqueeze(1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_maps_tensor, X_sources_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return model


def visualize_prediction(model, binary_map, source_x, source_y):
    """
    Visualizza predizione del modello vs simulazione ground truth
    """
    # Ground truth
    simulator = GaussianDispersionSimulator(binary_map.shape[0])
    true_concentration = simulator.simulate(binary_map, source_x, source_y)
    
    # Predizione modello
    model.eval()
    with torch.no_grad():
        map_tensor = torch.FloatTensor(binary_map).unsqueeze(0).unsqueeze(0)
        source_tensor = torch.FloatTensor([[source_x, source_y]])
        predicted_concentration = model(map_tensor, source_tensor)
        predicted_concentration = predicted_concentration.squeeze().numpy()
    
    # Visualizzazione
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Mappa edifici
    im1 = axes[0].imshow(binary_map, cmap='RdYlGn', origin='lower')
    axes[0].plot(source_x, source_y, 'ro', markersize=10, label='Sorgente')
    axes[0].set_title('Mappa Edifici + Sorgente')
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0])
    
    # Ground truth
    im2 = axes[1].imshow(true_concentration, cmap='hot', origin='lower')
    axes[1].set_title('Dispersione Simulata (Ground Truth)')
    plt.colorbar(im2, ax=axes[1])
    
    # Predizione
    im3 = axes[2].imshow(predicted_concentration, cmap='hot', origin='lower')
    axes[2].set_title('Dispersione Predetta (CNN)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Calcola errore
    mse = np.mean((true_concentration - predicted_concentration)**2)
    print(f"MSE tra predizione e ground truth: {mse:.6f}")


# Esempio di utilizzo completo
if __name__ == "__main__":
    # Carica la mappa di Benevento (generata dal tuo codice)
    binary_map = np.load("./GNN/benevento,_italy_full_map.npy")
    print(f"Mappa caricata: {binary_map.shape}")
    
    # Genera dataset di training
    X_maps, X_sources, y_concentrations = generate_training_dataset(
        binary_map, n_samples=500
    )
    
    # Crea e addestra il modello
    model = DispersionCNN(grid_size=binary_map.shape[0])
    print(f"Modello creato: {sum(p.numel() for p in model.parameters())} parametri")
    
    # Training
    trained_model = train_model(
        model, X_maps, X_sources, y_concentrations, epochs=30
    )
    
    # Test su esempio
    test_x, test_y = 150, 100  # Posizione test
    visualize_prediction(trained_model, binary_map, test_x, test_y)
    
    print("Training completato! Il modello ha imparato a predire la dispersione.")