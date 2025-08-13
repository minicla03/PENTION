import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
from scipy.interpolate import RegularGridInterpolator
from config import WindType, StabilityType, NPS, PasquillGiffordStability

class Sensor:
    def __init__(self, sensor_id, x: float, y: float, z: float = 2.,
                 noise_level: float = 0.1, is_fault: bool = False,
                 info: dict = {}):
        self.id = sensor_id
        self.x = x
        self.y = y
        self.z = z
        self.noise_level = noise_level
        self.is_fault = is_fault
        self.info = info 
        self.concentrations = None
        self.noisy_concentrations = None
        self.times= None

    def sample_substance(self, conc_field, x_grid, y_grid, t_grid):
        """
        Campiona il campo di concentrazione 3D (x, y, t) alla posizione del sensore.
        """

        if self.is_fault:
            print(f"Sensor {self.id} is faulty. No data sampled.")
            self.concentrations = np.array([], dtype=float)
            self.noisy_concentrations = np.array([], dtype=float)
            self.times = []
            return 
        
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

    def _sample_meteorology(self):
        wind_type = random.choice([WindType.CONSTANT ,WindType.PREVAILING, WindType.FLUCTUATING])
        stability_type = StabilityType.CONSTANT
        if stability_type == StabilityType.CONSTANT:
            stability_value = random.choice([
                PasquillGiffordStability.VERY_UNSTABLE,
                PasquillGiffordStability.MODERATELY_UNSTABLE,
                PasquillGiffordStability.SLIGHTLY_UNSTABLE,
                PasquillGiffordStability.NEUTRAL,
                PasquillGiffordStability.MODERATELY_STABLE,
                PasquillGiffordStability.VERY_STABLE
            ])
        else:
            # Fallback to a neutral stability ensuring the correct enum type
            stability_value = PasquillGiffordStability.NEUTRAL
        
        wind_speed = self._assign_wind_speed(stability_value)  
        humidify = random.choice([True, False])
        dry_size = 1.0
        RH = round(np.random.uniform(0, 0.99), 2) if humidify else 0.0

        return wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH

    
    def _assign_wind_speed(self, stability: PasquillGiffordStability) -> float:
        """
        Restituisce una velocità del vento (m/s) coerente con la stabilità atmosferica.
        I range sono basati su letteratura meteorologica semplificata.
        """
        if stability == PasquillGiffordStability.VERY_UNSTABLE:  # A
            return round(random.uniform(2.0, 6.0), 2)
        elif stability == PasquillGiffordStability.MODERATELY_UNSTABLE:  # B
            return round(random.uniform(2.0, 5.0), 2)
        elif stability == PasquillGiffordStability.SLIGHTLY_UNSTABLE:  # C
            return round(random.uniform(3.0, 6.5), 2)
        elif stability == PasquillGiffordStability.NEUTRAL:  # D
            return round(random.uniform(4.0, 8.0), 2)
        elif stability == PasquillGiffordStability.MODERATELY_STABLE:  # E
            return round(random.uniform(1.0, 4.0), 2)
        elif stability == PasquillGiffordStability.VERY_STABLE:  # F
            return round(random.uniform(0.5, 3.0), 2)
        else:
            return round(random.uniform(2.0, 6.0), 2)

    def _simulate_mass_spectrum(self):
        """
        Simula uno spettro di massa sintetico a partire dalle concentrazioni.
        """
        num_bins = 600
        np.random.seed(self.id)
        baseline = np.random.rand(num_bins) * 0.01
        peak_positions = np.random.choice(range(num_bins), size=3, replace=False)
        spectrum = baseline.copy()

        if self.noisy_concentrations is None or len(self.noisy_concentrations) == 0:
            mean_conc = 0.0
        else:
            mean_conc = float(np.mean(np.asarray(self.noisy_concentrations, dtype=float)))
        for pos in peak_positions:
            intensity_peak = mean_conc * (1 + np.random.rand())
            spectrum[pos] += intensity_peak
        if self.noise_level > 0:
            noise = baseline * (1 + np.random.rand(num_bins))
            spectrum += noise
        return spectrum

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
        
    def run_sensor(self):
        '''
        Esegue il campionamento del sensore.
        Campiona meteorologia, sostanza e simula lo spettro di massa.
        Se il sensore è in stato di guasto, non campiona dati.
        
        Returns:
            dict: Dati campionati dal sensore, inclusi tempi, spettro di massa,
                velocità del vento, tipo di vento, tipo di stabilità, valore di stabilità,
                umidità, dimensione secca e umidità relativa (RH).
        '''
        if self.is_fault is True:
            print(f"Sensor {self.id} is faulty. No data sampled.")
            self.concentrations = np.array([], dtype=float)
            self.noisy_concentrations = np.array([], dtype=float)
            self.times = []
            return
        
        wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH =self._sample_meteorology()
        mass_spectrum = [self._simulate_mass_spectrum() for _ in range(3)]
        
        data = {
            "times": self.times,
            "mass_spectrum": mass_spectrum,
            "wind_speed": wind_speed,
            "wind_type": wind_type,
            "stability_type": stability_type,
            "stability_value": stability_value,
            "humidify": humidify,
            "dry_size": dry_size,
            "RH": RH,
        }
        return data

