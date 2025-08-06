from config import ModelConfig, OutputType, WindType, StabilityType, NPS, PasquillGiffordStability
from gaussianModel import run_dispersion_model
import plot_utils
from scipy.interpolate import RegularGridInterpolator
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium

BINARY_MAP_PATH = os.path.join("GNN/binary_maps_data/benevento_italy_full_map.npy")
N_SENSORS = 10

# Caricamento mappa binaria (1 = suolo libero, 0 = edificio)
binary_map = np.load(BINARY_MAP_PATH)
free_cells = np.argwhere(binary_map == 1)

def random_position():
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(x), float(y)

def add_noise(concentrations, noise_level=0.05):
    concentrations = np.array(concentrations)
    noise_std = noise_level * np.maximum(concentrations, 1e-6)
    noise = np.random.normal(0, noise_std, size=concentrations.shape)
    noisy_conc = concentrations + noise
    return np.clip(noisy_conc, 0, None)

if __name__ == "__main__":

    origin_lat_ref, origin_lon_ref = 41.1305, 14.7826  # lat/lon di riferimento della mappa

    origin_x, origin_y = random_position()  # coordinate in metri nella griglia

    origin_lat, origin_lon = plot_utils.meters_to_latlon(origin_x, origin_y, origin_lat_ref, origin_lon_ref)

    # Posizioni dei sensori (x, y)
    sensors = [random_position() for _ in range(N_SENSORS)]

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
        stacks=[(origin_lat, origin_lon, 0.0020, 4.68)],
        dry_size=1.0,
        x_slice=26,
        y_slice=1,
        sensor_locations=[]
    )

    result = run_dispersion_model(config)
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label = result

    noise_level = 0.004
    concentrations_noisy = add_noise(C1, noise_level)

    x_sorted = np.sort(np.unique(x))
    y_sorted = np.sort(np.unique(y))
    times = np.sort(np.unique(times))

    C_sorted = concentrations_noisy

    plot_utils.plot_surface_view_3d(C1, x_sorted, y_sorted, z=z, times=times, 
                              source=(origin_lat,origin_lon), sensors=sensors,
                              t_index=None, z_index=0,
                              title="Visualizzazione 3D con overlay sensori/sorgente")
    # Interpolazione regolare per valori di concentrazione ai sensori
    interp_func = RegularGridInterpolator((x_sorted, y_sorted, times), C_sorted, bounds_error=False, fill_value=0.0)

    # Coordinate geografiche dei sensori
    sensor_locs_geo = [plot_utils.meters_to_latlon(x_s, y_s, origin_lat, origin_lon) for x_s, y_s in sensors]

    # Calcolo concentrazioni medie interpolate per ogni sensore
    sensor_concentrations = []
    for (sensor_x, sensor_y), (lat, lon) in zip(sensors, sensor_locs_geo):
        conc_time = np.array([interp_func((sensor_x, sensor_y, t)) for t in times])
        mean_conc = np.mean(conc_time)
        sensor_concentrations.append((lat, lon, mean_conc))
        plot_utils.plot_timeseries(times, conc_time, sensor_id=(sensor_x, sensor_y))

    # Normalizza i valori per la mappa colori
    concs = [c for (_, _, c) in sensor_concentrations]
    norm = mcolors.Normalize(vmin=min(concs), vmax=max(concs))
    cmap = cm.get_cmap('coolwarm')  # blu -> rosso

    # Crea la mappa base (media temporale)
    mappa = plot_utils.plot_puff_on_map(C1, x, y, center_lat=origin_lat, center_lon=origin_lon)

    # Aggiungi marker colorati per i sensori
    for sid, (lat, lon, conc) in enumerate(sensor_concentrations):
        color = mcolors.to_hex(cmap(norm(conc)))
        folium.CircleMarker(
            location=(lat, lon),
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.9,
            fill_color=color,
            popup=folium.Popup(f"<b>Sensore {sid}</b><br>Conc. media: {conc:.2e} µg/m³", max_width=200)
        ).add_to(mappa)

    # Salva mappa con marker colorati
    mappa.save("gaussianPuff/example/puff_map_with_sensors_3007_colored.html")
    print("📍 Mappa salvata: puff_map_with_sensors_3007_colored.html")

    # Mappa con dati rumorosi (noisy)
    mappa_noisy = plot_utils.plot_puff_on_map(concentrations_noisy, x, y, center_lat=origin_lat, center_lon=origin_lon, sensor_locs=sensor_locs_geo)
    mappa_noisy.save("gaussianPuff/example/puff_map_with_sensors_3007_noisy.html")
    print("📍 Mappa rumore salvata: puff_map_with_sensors_3007_noisy.html")

    # Visualizza grafico statico plan view (media temporale)
    label_str = (stab_label or "") + '\n' + (wind_label or "")
    plot_utils.plot_plan_view(C1, x, y, label_str)