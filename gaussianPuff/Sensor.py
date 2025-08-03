import numpy as np

class Sensor:
    def __init__(self, name,  x, y, z, sensor_id, noise_level=0.1):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.sensor_id = sensor_id
        self.noise_level = noise_level
        self.measurements = []

    def simulate_sensor_measurements(self, time_points, total_concentration):
        """
        Simula le misurazioni dei sensori aggiungendo rumore realistico
        """
        for t in time_points:
            true_conc = total_concentration[t]
            # Aggiunge rumore gaussiano
            noise = np.random.normal(0, self.noise_level * max(true_conc, 0.01))
            measured_conc = max(0, true_conc + noise)
            self.measurements.append({
                'time': t,
                'concentration': measured_conc,
                'true_concentration': true_conc
            })
    
    def __repr__(self):
        return f"Sensor(name={self.name}, id={self.sensor_id}, location=({self.x}, {self.y}, {self.z}), noise_level={self.noise_level})"
    

