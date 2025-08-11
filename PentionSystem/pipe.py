import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gaussianPuff')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CorrectionDispersion')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EmissionSourceLocalization')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ClassificatoreNPS')))

from gaussianPuff.Sensor import Sensor
from gaussianPuff.config import PasquillGiffordStability, WindType, StabilityType
from ClassificatoreNPS.pipe_clf import pipe_clf_dnn, pipe_clf_brf

    
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-08-08.csv")

if __name__ == "__main__":

    binary_map = np.load(BINARY_MAP_PATH)
    csv_df = pd.read_csv(CSV_PATH)
    csv_df = csv_df.drop(labels=['aerosol_type','source_x', 'source_y','source_h' ])

    #istnazione dei sensori

    #spettri
    spectra

    #classificazione
    predictions=pipe_clf_brf(spectra)
    predictions=pipe_clf_dnn(spectra)
