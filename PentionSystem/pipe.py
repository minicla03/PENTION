import os
import sys
import numpy as np
import random
import pandas as pd
import threading as thr
from typing import Union, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gaussianPuff')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CorrectionDispersion')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EmissionSourceLocalization')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ClassificatoreNPS')))

from gaussianPuff.Sensor import Sensor
from gaussianPuff.config import PasquillGiffordStability, WindType, StabilityType, ModelConfig, NPS, OutputType, DispersionModelType, ConfigPuff
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.plot_utils import plot_plan_view
from ClassificatoreNPS.pipe_clf import pipe_clf_dnn, pipe_clf_brf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-08-08.csv")

N_SENSORS = 10
BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")

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

def fault_probability(wind_speed, stability_value, RH, wind_type):
    # Base probability
    base_prob = 0.1
    
    # Aumenta probabilità con vento forte
    if wind_speed > 6.0:
        base_prob += 0.2
    
    # Aumenta probabilità se stabilità molto instabile o molto stabile
    if stability_value in [PasquillGiffordStability.VERY_UNSTABLE, PasquillGiffordStability.VERY_STABLE]:
        base_prob += 0.15
        
    # Aumenta probabilità con alta umidità
    if RH > 0.8:
        base_prob += 0.2
        
    # Vento fluttuante aumenta la probabilità
    if wind_type == WindType.FLUCTUATING:
        base_prob += 0.1
        
    # Limita la probabilità al massimo di 0.75 per evitare valori troppo estremi
    return min(base_prob, 0.75)

nps_type_map = {
    "CANNABINOID_ANALOGUES": 0,
    "CATHINONE_ANALOGUES": 1,
    "PHENETHYLAMINE_ANALOGUES": 2,
    "PIPERAZINE_ANALOGUES": 3,
    "TRYPTAMINE_ANALOGUES": 4,
    "FENTANYL_ANALOGUES": 5,
    "OTHER_COMPOUNDS": 6
}
idx_to_nps = {v: k for k, v in nps_type_map.items()}

def normalize_prediction(pred: Union[int, str]) -> str:
    if isinstance(pred, int):
        return idx_to_nps.get(pred, "OTHER_COMPOUNDS")
    if isinstance(pred, str):
        # normalizza maiuscole e spazi/underscore
        key = pred.strip().upper().replace(" ", "_")
        return key if key in nps_type_map else "OTHER_COMPOUNDS"
    return "OTHER_COMPOUNDS"

if __name__ == "__main__":

    binary_map = np.load(BINARY_MAP_PATH)
    csv_df = pd.read_csv(CSV_PATH)
    csv_df = csv_df.drop(labels=['aerosol_type','source_x', 'source_y','source_h' ])

    #meteorologia
    wind_speed, wind_type, stab_type, stab_value = sample_meteorology()

    #istanziazione dei sensori
    sensors = []
    for j in range(N_SENSORS):
        x, y=random_position()

        fault_prob = fault_probability(wind_speed, stab_value, 0.5, wind_type) 
        is_fault = random.random() < fault_prob

        sensor = Sensor(
            j,
            x=x,
            y=y,
            z=2.0,
            noise_level=round(np.random.uniform(0.0, 0.0005), 4),
            is_fault=is_fault
        )
        sensors.append(sensor)

    #attivazione dei sensori
    data_records: List[dict] = []
    for s in sensors:
        rec = s.run_sensor() 
        if getattr(s, "is_fault", False) and "mass_spectrum" in rec:
            data_records.append(rec)
        
    #classificazione
    predictions_brf = []
    predictions_dnn = []
    for dict in data_records:
        predictions_brf=pipe_clf_brf(dict["mass_spectrum"])
        predictions_dnn=pipe_clf_dnn(dict["mass_spectrum"])

    print("Predizioni BRF:", predictions_brf)
    print("Predizioni DNN:", predictions_dnn)

    #nps = [pred for pred in predictions_brf if pred in nps_type_map]

    #indentificazione della posizione della sorgente
    

    #modello gaussiano della dispersione
    """x_slice=26
    y_slice=26

    config = ModelConfig(
        days=8,
        RH=0.01,
        aerosol_type=NPS.TRYPTAMINE_ANALOGUES,
        humidify=True,
        stability_profile=StabilityType.CONSTANT,
        stability_value=PasquillGiffordStability.MODERATELY_UNSTABLE,
        wind_type=WindType.FLUCTUATING,
        wind_speed=10.,
        output=OutputType.PLAN_VIEW,
        stacks=[(origin_lat, origin_lon, 0.0020, 4.68)],
        dry_size=1.0,
        x_slice=x_slice,
        y_slice=y_slice,
        #grid_size=binary_map_metadata['grid_size'][0],
        dispersion_model=DispersionModelType.PLUME,
        config_puff=ConfigPuff(puff_interval=1, max_puff_age=6)  # 1 hour interval, max age 6 hours
    )

    result = run_dispersion_model(config)#, bounds=binary_map_metadata['bounds'])
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = result
    
    plot_plan_view(C1, x, y, f"Plan View - {stab_label} - {wind_label}", wind_dir, 10. ,puff, stability_class=PasquillGiffordStability.NEUTRAL.value)"""
    
    #perfezionamento della diffusione 
