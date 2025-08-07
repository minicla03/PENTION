import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class Sensor:
    def __init__(self, sensor_id, x: float, y: float, z: float =2., noise_level: float=0.1):
        self.id = sensor_id
        self.x = x
        self.y = y
        self.z = z
        self.noise_level = noise_level
        self.concentrations = None
        self.noisy_concentrations = None
        self.times = None

    def sample(self, conc_field, x_grid, y_grid, t_grid):
        """
        Campiona il campo di concentrazione 3D (x, y, t) alla posizione del sensore.
        """
        x_sorted = np.sort(np.unique(x_grid))
        y_sorted = np.sort(np.unique(y_grid))
        times = np.sort(np.unique(t_grid))
        self.times = times

        interpolator = RegularGridInterpolator((x_sorted, y_sorted, times), conc_field, bounds_error=False, fill_value=0.0)
        coords = [(self.x, self.y, t) for t in times]
        self.concentrations = np.array([interpolator(c) for c in coords])

        if self.noise_level > 0.0:
            noise_std = self.noise_level * np.maximum(self.concentrations, 1e-6)
            noise = np.random.normal(0, noise_std)
            self.noisy_concentrations = np.clip(self.concentrations + noise, 0, None)
        else:
            self.noisy_concentrations = self.concentrations.copy()

    def simulate_mass_spectrum(self):
        """
        Simula uno spettro di massa sintetico a partire dalle concentrazioni.
        Restituisce un vettore di intensità (es. 128 valori fittizi su range m/z 0–600).
        """
        num_bins = 128
        np.random.seed(self.id)  

        # Mappa fittizia: concentrazioni → intensità su m/z
        baseline = np.random.rand(num_bins) * 0.01
        peak_positions = np.random.choice(range(0, 600), size=3, replace=False)
        spectrum = baseline.copy()
        for pos in peak_positions:
            intensity = np.mean(self.noisy_concentrations) * (1 + np.random.rand())
            spectrum[pos] += intensity

        return spectrum

    def classify_NPS(self, classifier_model, specturum):
      
        prediction = classifier_model.predict(specturum)
        detected_class = prediction[0]
        return detected_class

    def plot_timeseries(self, use_noisy=True):
        if self.times is None:
            raise ValueError("Il sensore non ha ancora campionato dati.")

        data = self.noisy_concentrations if use_noisy else self.concentrations

        if data is None:
            raise ValueError("I dati di concentrazione non sono disponibili per il sensore.")
        
        plt.plot(self.times, data, label=f"Sensor {self.id}")
        plt.xlabel("Tempo (h)")
        plt.ylabel("Concentrazione [μg/m³]")
        plt.title(f"Andamento temporale - Sensore {self.id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_data(self):
        return self.times, self.concentrations, self.noisy_concentrations
