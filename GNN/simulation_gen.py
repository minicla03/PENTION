import numpy as np
import pandas as pd
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gaussianPuff')))
from config import ModelConfig, StabilityType, WindType, OutputType, NPS, PasquillGiffordStability, DispersionModelType, ConfigPuff
from gaussianModel import run_dispersion_model
import plot_utils
from scipy.interpolate import RegularGridInterpolator

# Parametri generali
N_SIMULATIONS = 1000
N_SENSORS = 10
SAVE_DIR = "./GNN/dataset"
SAVE_DIR_CONC_REAL= "./GNN/dataset/real_dispersion"
BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")

os.makedirs(SAVE_DIR, exist_ok=True)

# Caricamento mappa binaria (1 = suolo libero, 0 = edificio)
binary_map = np.load(BINARY_MAP_PATH)
free_cells = np.argwhere(binary_map == 1)

def random_position():
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(x), float(y)

def assign_wind_speed(stability: PasquillGiffordStability) -> float:
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
    
def plot_concentrazione_su_mappa(concentrazione, binary_map, t):
    """
    Visualizza la concentrazione al tempo t sulla mappa, evidenziando gli edifici in nero.
    
    Args:
        concentrazione (np.ndarray): array 3D (nx, ny, nt)
        binary_map (np.ndarray): array 2D binaria (nx, ny) con 0 edificio, 1 libero
        t (int): indice temporale
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    C_frame = concentrazione[:, :, t]
    C_norm = C_frame / (C_frame.max() if C_frame.max() > 0 else 1)

    # Crea immagine RGB bianca
    img_rgb = np.ones(C_frame.shape + (3,))

    # Colormap viridis per concentrazione > 0
    colored = cm.viridis(C_norm)
    img_rgb[C_frame > 0] = colored[C_frame > 0, :3]

    # Edifici in nero dove concentrazione == 0
    mask_edifici = (C_frame == 0) & (binary_map == 0)
    img_rgb[mask_edifici] = [0, 0, 0]

    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb, origin='lower')
    plt.title(f"Concentrazione e edifici (tempo {t})")
    plt.axis('off')
    plt.show()

def sample_meteorology():
    wind_type = random.choice([WindType.CONSTANT ,WindType.PREVAILING, WindType.FLUCTUATING])
    stability_type = StabilityType.CONSTANT
    stability_value =  random.choice([PasquillGiffordStability.VERY_UNSTABLE, 
                                      PasquillGiffordStability.MODERATELY_UNSTABLE, 
                                      PasquillGiffordStability.SLIGHTLY_UNSTABLE, 
                                      PasquillGiffordStability.NEUTRAL,  
                                      PasquillGiffordStability.MODERATELY_STABLE, 
                                      PasquillGiffordStability.VERY_STABLE]) if stability_type == StabilityType.CONSTANT else 0
    
    wind_speed = assign_wind_speed(stability_value) # type: ignore

    return wind_speed, wind_type, stability_type, stability_value

def add_noise(concentrations, noise_level):
    concentrations = np.array(concentrations)
    noise_std = noise_level * np.maximum(concentrations, 1e-6)
    noise = np.random.normal(0, noise_std)
    noisy_conc = concentrations + noise
    noisy_conc = np.clip(noisy_conc, 0, None)
    return noisy_conc.tolist()

data_records = []

for i in range(N_SIMULATIONS):
    print(f"Simulazione {i+1}/{N_SIMULATIONS}")

    # Sorgente (stack): posizione, altezza, emissione
    x_src, y_src = random_position()
    h_src = round(np.random.uniform(1, 10), 2)  # altezza del pennacchio 
    Q = round(np.random.uniform(0.0001, 0.01), 4)  # tasso di emissione 
    stacks = [(x_src, y_src, Q, h_src)]

    # Sensori
    sensors = [random_position() for _ in range(N_SENSORS)]

    # Meteo
    wind_speed, wind_type, stab_type, stab_value = sample_meteorology()

    # nps considerato casuale
    aerosol_type = random.choice(list(NPS))

    humidify = random.choice([True, False])

    config = ModelConfig(
        days=20,
        aerosol_type=aerosol_type,
        dry_size=1.0,
        humidify=humidify,
        RH=round(np.random.uniform(0, 0.99),2) if humidify else 0.0,
        stability_profile=stab_type,
        stability_value=stab_value, # type: ignore
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        x_slice=26,
        y_slice=1,
        dispersion_model=DispersionModelType.PLUME,
    )

    # Calcola concentrazioni con modello gaussiano
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = run_dispersion_model(config)
    plot_utils.plot_plan_view(C1, x, y, f"Plan View - {stab_label} - {wind_label}", puff, stability_class=PasquillGiffordStability.NEUTRAL.value)
    binary_map=binary_map[:,:, np.newaxis]  # Aggiungi una dimensione per la compatibilità con C1
    mask_edifici = (binary_map == 1) & (C1 != 0)
    #img_rgb[mask_edifici] = [0, 0, 0]
    plot_utils.plot_plan_view(mask_edifici, x, y, f"Plan View - {stab_label} - {wind_label}")
    #plot_concentrazione_su_mappa(C1, binary_map, t=0)  # Visualizza al tempo 0

    
    print(type(x), x.shape)
    print(type(y), y.shape)
 
    # Aggiungi rumore simulato
    noise_level = round(np.random.uniform(0.0, 0.0005), 4)
    concentrations_noisy = np.array([add_noise(c, noise_level) for c in C1])

    print(type(concentrations_noisy), concentrations_noisy.shape)
    print(type(concentrations_noisy[0]))
    print(type(concentrations_noisy[0][0][0]))

    filename = f"sim_{i}_conc_real_0408_v3.npy"
    np.save(os.path.join(SAVE_DIR_CONC_REAL, filename), concentrations_noisy)

    x_sorted = np.sort(np.unique(x))
    y_sorted = np.sort(np.unique(y))
    times = np.sort(np.unique(times))
    # Riordina i dati lungo gli assi corrispondenti
    C_sorted = concentrations_noisy

    # Interpolazione regolare per ottenere valori di concentrazione nei punti dei sensori
    interp_func = RegularGridInterpolator((x_sorted, y_sorted, times), C_sorted, bounds_error=False, fill_value=0.0)

    # Salva record per ogni sensore e ogni simulazione
    for sid, (sensor_x, sensor_y) in enumerate(sensors):

        # Converti le coordinate reali sensore (sensor_x, sensor_y) in coordinate di griglia (ad esempio intorno alla cella più vicina)
        sensor_x_grid = int(np.clip(sensor_x, 0, C1.shape[0] - 1))
        sensor_y_grid = int(np.clip(sensor_y, 0, C1.shape[1] - 1))

        conc= np.array([interp_func((sensor_x, sensor_y, t)) for t in times])
        print(type(conc), conc.shape)
        #plot_timeseries(times, conc, sensor_id=sid)
        
        row = {
            "simulation_id": i,
            "sensor_id": sid,
            "sensor_x": sensor_x,
            "sensor_y": sensor_y,
            "sensor_noise": noise_level,
            "sensor_height": 2.0,  # altezza sensore fissa
            "days": config.days,
            "RH": config.RH,
            "humidify": config.humidify,
            #"timestamp": datetime.now().isoformat(),
            "wind_type": wind_type.name,
            "wind_speed": wind_speed,
            "wind_dir": ",".join(map(str, wind_dir.tolist())),
            "stability_profile": stab_type.name,
            "stability_value": stab_value,
            "aerosol_type": aerosol_type.name,
            "source_x": x_src,
            "source_y": y_src,
            "source_h": h_src,
            "emission_rate": Q,
            "real_concentration": filename,
            "contratio_series": ",".join(map(str, conc))
        }
        #for t_idx, c_val in enumerate(conc):
        #    row[f"c_t{t_idx}"] = c_val

        data_records.append(row)

# Salvataggio CSV
df = pd.DataFrame(data_records)
csv_path = os.path.join(SAVE_DIR, "nps_simulated_dataset_gaussiano_0608.csv")
df.to_csv(csv_path, index=False)

print(f"\nDataset generato e salvato in {csv_path}")
