import numpy as np
import joblib

mz_range = np.arange(1, 601)

def _compute_features(spectrum):
    peaks = [(mz, intensity) for mz, intensity in zip(mz_range, spectrum) if intensity > 0]
    if not peaks:
        return [np.nan] * 13

    mz_values, intensities = zip(*peaks)
    mz_values = np.array(mz_values)
    intensities = np.array(intensities)

    # Base Peak Mass
    base_peak_idx = np.argmax(intensities)
    base_peak_mass = mz_values[base_peak_idx]
    
    # Base Peak Proximity
    if len(mz_values) > 1:
        mz_diff_base = np.abs(mz_values - base_peak_mass)
        base_prox = np.partition(mz_diff_base[mz_diff_base != 0], 0)[0]
    else:
        base_prox = 0.0

    # Maximum Mass
    max_mass = np.max(mz_values)

    # Maximum Mass Proximity
    if len(mz_values) > 1:
        mz_diff_max = np.abs(mz_values - max_mass)
        max_prox = np.partition(mz_diff_max[mz_diff_max != 0], 0)[0]
    else:
        max_prox = 0.0

    # Number of Peaks
    num_peaks = len(peaks)

    # Intensity stats
    intensity_mean = np.mean(intensities)
    intensity_std = np.std(intensities)
    intensity_density = np.max(intensities) / num_peaks

    # Mass stats
    mass_mean = np.mean(mz_values)
    mass_std = np.std(mz_values)
    mass_density = max_mass / num_peaks

    # Pairwise Peak Mass Differences
    diffs = np.abs(np.subtract.outer(mz_values, mz_values))
    diffs = diffs[np.triu_indices(len(diffs), k=1)]
    diff_counts = np.bincount(np.round(diffs).astype(int))
    ppmd = np.argmax(diff_counts)
    mean_ppmd = np.mean(diffs)

    return [
        base_peak_mass, base_prox, max_mass, max_prox,
        num_peaks, intensity_mean, intensity_std, intensity_density,
        mass_mean, mass_std, mass_density, ppmd, mean_ppmd
    ]

def pipe_clf_dnn(spectra):
    dnn_clf= joblib.load('model/dnn_spectra_version.keras')
    predictions = []
    for spectrum in spectra:
        prediction = dnn_clf.predict(spectrum)
        predictions.append(prediction[0])
    return np.array(predictions)

def pipe_clf_brf(spectra):
    brf_clf = joblib.load('model/balanced_random_forest_brf.pkl')
    predictions = []
    for spectrum in spectra:
        features = np.array(_compute_features(spectrum))
        prediction = brf_clf.predict(features)
        predictions.append(prediction[0])
    return np.array(predictions)