from config import ModelConfig, OutputType, WindType, StabilityType, NPS, PasquillGiffordStability, DispersionModelType, ConfigPuff
from gaussianModel import run_dispersion_model
import plot_utils
import os
from Sensor import Sensor
import numpy as np


BINARY_MAP_PATH = os.path.join("GNN/binary_maps_data/roma_italy_bbox.npy")
N_SENSORS = 5

# Caricamento mappa binaria (1 = suolo libero, 0 = edificio)
binary_map = np.load(BINARY_MAP_PATH)
free_cells = np.argwhere(binary_map == 1)

def random_position():
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(x), float(y)

if __name__ == "__main__":

    origin_lat_ref, origin_lon_ref = 41.1305, 14.7826  # lat/lon di riferimento della mappa

    origin_x, origin_y = random_position()  # coordinate in metri nella griglia

    origin_lat, origin_lon = plot_utils.meters_to_latlon(origin_x, origin_y, origin_lat_ref, origin_lon_ref)

    sensors = []
    for i in range(N_SENSORS):
        x, y=random_position()
        sensor = Sensor(i, x=x ,y=y, z=2.0, noise_level=round(np.random.uniform(0.0, 0.0005), 4))
        sensors.append(sensor)

    x_slice=26
    y_slice=1

    config = ModelConfig(
        days=8,
        RH=0.65,
        aerosol_type=NPS.TRYPTAMINE_ANALOGUES,
        humidify=True,
        stability_profile=StabilityType.ANNUAL,
        stability_value=PasquillGiffordStability.SLIGHTLY_UNSTABLE,
        wind_type=WindType.PREVAILING,
        wind_speed=4.94,
        output=OutputType.PLAN_VIEW,
        stacks=[(origin_lat, origin_lon, 0.0020, 4.68)],
        dry_size=1.0,
        x_slice=x_slice,
        y_slice=y_slice,
        dispersion_model=DispersionModelType.PLUME,
        config_puff=ConfigPuff(puff_interval=1, max_puff_age=6)  # 1 hour interval, max age 6 hours
    )

    result = run_dispersion_model(config)
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = result
    
    plot_utils.plot_plan_view(C1, x, y, f"Plan View - {stab_label} - {wind_label}", puff, stability_class=PasquillGiffordStability.NEUTRAL.value)
    binary_map=binary_map[:,:, np.newaxis]  # Aggiungi una dimensione per la compatibilità con C1
    mask_edifici = (binary_map == 1) & (C1 != 0)
    plot_utils.plot_plan_view(mask_edifici, x, y, f"Plan View - {stab_label} - {wind_label}")
    plot_utils.plot_surface_time(C1, times, x_slice, y_slice, stability, stab_label, f"Surface Time - {stab_label} - {wind_label}")
    plot_utils.plot_height_slice(C1, y, z, stab_label, wind_label)

    for sensor in sensors:
        sensor.sample(C1, x, y, times)
        sensor.plot_timeseries(use_noisy=True)
    
    sensor_data = [(sensor.x, sensor.y) for sensor in sensors]

    mappa = plot_utils.plot_concentration_with_sensors(C1, x, y, sensor_data, (origin_lat, origin_lon), times, title="Concentrazione con sensori")
    if mappa is not None:
        mappa.save("gaussianPuff/example/puff_map_with_sensors_0708.html")
        print("📍 Mappa salvata: puff_map_with_sensors_0708.html")
    else:
        print("⚠️ Errore: la funzione plot_concentration_with_sensors ha restituito None, impossibile salvare la mappa.")
    
    fig, ax = plot_utils.plot_plan_view_with_mask(C1, x, y, binary_map, title="Plan View with Mask")
    fig.savefig("gaussianPuff/example/puff_plan_view_with_mask_0708.png")
