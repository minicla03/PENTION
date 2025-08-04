# file: main.py
from config import ModelConfig, OutputType, WindType, StabilityType, NPS, PasquillGiffordStability
from gaussianModel import run_dispersion_model
from plot_utils import (
    plot_plan_view, plot_surface_time, plot_puff_on_map, 
    meters_to_latlon, plot_height_slice
)
import numpy as np
from Sensor import Sensor

def add_noise(concentrations, noise_level=0.05):
    concentrations = np.array(concentrations)
    noise_std = noise_level * np.maximum(concentrations, 1e-6)
    noise = np.random.normal(0, noise_std, size=concentrations.shape)
    noisy_conc = concentrations + noise
    return np.clip(noisy_conc, 0, None)
    #return noisy_conc.tolist()

if __name__ == "__main__":

    origin_lat, origin_lon = 41.1305, 14.7826

    # Posizioni dei sensori (x, y)
    sensori = [
        Sensor(name="Sensor_1", x=100, y=200, z=0, sensor_id=1, noise_level=0.05),
        Sensor(name="Sensor_2", x=300, y=150, z=0, sensor_id=2, noise_level=0.1),
        Sensor(name="Sensor_3", x=250, y=400, z=0, sensor_id=3, noise_level=0.07),
        Sensor(name="Sensor_4", x=500, y=300, z=0, sensor_id=4, noise_level=0.05),
    ]

    sensor_pos= [
        (s.x, s.y) for s in sensori
    ]

    sensor_locs_geo = [
        meters_to_latlon(x, y, origin_lat, origin_lon)
        for (x, y) in [(s.x, s.y) for s in sensori]
    ]

    """config = ModelConfig(
        days=10,
        RH=0.40,
        aerosol_type=NPS.CANNABINOID_ANALOGUES,
        dry_size=60e-9,
        humidify=True,
        stability_profile=StabilityType.CONSTANT,
        stability_value=PasquillGiffordStability.MODERATELY_UNSTABLE,
        wind_type=WindType.CONSTANT,
        wind_speed=10.,
        output=OutputType.PLAN_VIEW,
        stacks=[(0, 0, 50, 30)],  # (x, y, Q, H)
        x_slice=26,
        y_slice=12,
    )"""

    config = ModelConfig(
        days=8,
        RH=0.65,
        aerosol_type=NPS.TRYPTAMINE_ANALOGUES,
        humidify=True,
        stability_profile=StabilityType.CONSTANT,
        stability_value=PasquillGiffordStability.SLIGHTLY_UNSTABLE,
        wind_type=WindType.PREVAILING,
        wind_speed=4.94,
        output=OutputType.PLAN_VIEW,
        stacks=[(40.0, 15.0, 0.0020, 4.68)],
        dry_size=1.0,
        x_slice=26,
        y_slice=1,
        sensor_locations=[]
    )


    result = run_dispersion_model(config)
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label = result
    print(C1)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(wind_dir)
    noise_level=0.004 #round(np.random.uniform(0.0, 0.5), 2)
    print(noise_level)
    concentrations_noisy = add_noise(C1, noise_level)
    
    # Visualizza risultati

    label_str = (stab_label or "") + '\n' + (wind_label or "")
    plot_plan_view(C1, x, y, label_str)
    plot_plan_view(concentrations_noisy, x, y, label_str)
    
    plot_surface_time(C1, times, config.x_slice, config.y_slice, stability, stab_label, wind_label)
    
    plot_height_slice(C1, y, z, stab_label, wind_label)

    mappa = plot_puff_on_map(C1, x, y, center_lat=41.1305, center_lon=14.7826, sensor_locs=sensor_locs_geo)
    mappa_noisy= plot_puff_on_map(concentrations_noisy, x, y, center_lat=41.1305, center_lon=14.7826, sensor_locs=sensor_locs_geo)
    mappa.save("puff_map_with_sensors_3007.html")
    mappa.save("puff_map_with_sensors_3007_noisy.html")
    print("🛰️ Mappa interattiva salvata con i sensori: puff_map_with_sensors_3007.html")
    
