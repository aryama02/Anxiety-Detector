"""Acoustic feature extraction shared by training (train_daic.py) and the live app.

Changing the features, their order, or their count invalidates any saved model:
retrain with `python src/train_daic.py` afterwards.
"""

import librosa
import numpy as np


def extract_features(audio, sr=16000):
    """Return (vector, names, dict) for 39 acoustic features of a mono clip."""
    features = {}

    # 1. MFCCs (vocal tract shape / tension)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc))

    # 2. Pitch (F0) and jitter
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

    if len(f0_voiced) > 1:
        features["f0_mean"] = float(np.mean(f0_voiced))
        features["f0_std"] = float(np.std(f0_voiced))
        features["f0_range"] = float(np.max(f0_voiced) - np.min(f0_voiced))
        features["jitter"] = float(
            np.mean(np.abs(np.diff(f0_voiced))) / (np.mean(f0_voiced) + 1e-8)
        )
    else:
        features["f0_mean"] = features["f0_std"] = 0.0
        features["f0_range"] = features["jitter"] = 0.0

    # 3. Energy and shimmer
    rms = librosa.feature.rms(y=audio)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["shimmer"] = (
        float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-8)) if len(rms) > 1 else 0.0
    )

    # 4. Spectral shape and pause dynamics
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))
    features["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))

    features["voiced_ratio"] = (
        float(np.sum(voiced_flag) / len(voiced_flag)) if voiced_flag is not None else 0.0
    )
    is_silent = rms < 0.02
    pause_transitions = np.diff(is_silent.astype(int))
    features["num_pauses"] = float(np.sum(pause_transitions == 1))
    features["pause_ratio"] = float(np.sum(is_silent) / len(is_silent))

    return np.array(list(features.values())), list(features.keys()), features
