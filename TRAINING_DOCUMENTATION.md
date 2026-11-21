# Anxiety Detector Training Documentation

## Purpose

This guide documents how the acoustic XGBoost model was trained, what data was used, how the DAIC-WOZ files were processed, and how to retrain the model.

The training entry point is [src/train_daic.py](src/train_daic.py).

The live app consumes the output at:

```text
models/saved_voice_model.pkl
```

## Data Location

Place DAIC-WOZ data under:

```text
data/daic_data/
```

The script recognizes this location directly and also supports a top-level `daic_data/` folder.

The directory may contain:

- Participant ZIP archives.
- Extracted `_AUDIO.wav` files.
- Extracted `_TRANSCRIPT.csv` files.
- Score spreadsheets.
- Heavy feature and video files.

Access to DAIC-WOZ remains subject to the dataset's license and usage terms.

## Training Pipeline

The training process performs these steps:

1. Delete heavy unused files.
2. Selectively extract audio and transcript archives.
3. Merge participant score CSVs.
4. Match participant audio, transcripts, and labels.
5. Keep only participant speech.
6. Remove greeting speech.
7. Concatenate substantive participant segments.
8. Generate three 20-second chunks where possible.
9. Extract 39 acoustic features.
10. Train XGBoost with class weighting.
11. Evaluate with StratifiedKFold ROC-AUC.
12. Save a compressed model payload.

## Automatic Disk Cleanup

`prepare_daic_folders()` removes files matching:

- `*CLNF*`
- `*.bin`
- `*.mp4`
- `*.avi`
- `*FORMANT*`
- `*COVAREP*`

This frees disk space while preserving:

- `_AUDIO.wav`
- `_TRANSCRIPT.csv`
- CSV score spreadsheets.

## Selective ZIP Extraction

The script opens every `.zip` under `data/daic_data/` and extracts only files ending with:

- `_AUDIO.wav`
- `_TRANSCRIPT.csv`

The result is organized as:

```text
data/daic_data/<participant_id>/<participant_id>_AUDIO.wav
data/daic_data/<participant_id>/<participant_id>_TRANSCRIPT.csv
```

Video, 3D features, COVAREP, and FORMANT files inside ZIP archives are not extracted.

If six ZIPs in the dataset contain placeholder or invalid data, they are reported as unreadable rather than extracted.

## Score CSV Merging

`load_all_score_maps()` scans every root-level and nested CSV except `_TRANSCRIPT.csv`.

For each spreadsheet it detects:

- Any column containing `id` as the participant-ID column.
- Any column containing `phq` but not `binary` as the continuous score column.

Every valid row is merged into:

```python
score_map[participant_id] = phq8_score
```

Supported examples include:

- `labels.csv`
- `train_split_Depression_AVEC2017.csv`
- `dev_split_Depression_AVEC2017.csv`
- `test_split_Depression_AVEC2017.csv`
- `full_test_split.csv`

Later valid scores fill the same participant ID if another spreadsheet contains it.

## Transcript Processing

`get_participant_audio_chunks()` handles both tab-separated and comma-separated transcripts.

It:

1. Normalizes column names.
2. Keeps rows where `speaker.strip().lower() == "participant"`.
3. Skips the first two participant rows if there are more than four participant rows.
4. Ignores invalid start/stop values.
5. Keeps segments inside the audio duration.
6. Concatenates the remaining participant segments.
7. Normalizes volume.

The Ellie interviewer segments and virtual-interviewer speech are excluded.

## Multi-Chunk Augmentation

The concatenated participant audio is divided into up to three 20-second chunks:

```text
Chunk length: 20 seconds
Maximum chunks: 3
Minimum final chunk: 8 seconds
Short final chunk: zero-padded to 20 seconds
```

The intent is to expand the number of usable training samples without inventing synthetic audio.

## 39-Feature Extraction

Each 20-second chunk is passed to [src/aria/features.py](src/aria/features.py).

The shared function returns:

```python
vector, feature_names, feature_dictionary
```

The vector contains:

- 26 MFCC mean and standard-deviation values.
- F0 mean.
- F0 standard deviation.
- F0 range.
- Jitter.
- RMS mean.
- RMS standard deviation.
- Shimmer.
- Spectral centroid mean.
- Spectral centroid standard deviation.
- Zero-crossing rate.
- Voiced ratio.
- Pause count.
- Pause ratio.

The live app uses this exact same function. If feature extraction is modified, the saved model must be retrained.

## Binary Labeling

The continuous PHQ-8 score becomes a binary target:

```text
PHQ-8 >= 10  -> 1
PHQ-8 < 10   -> 0
```

## NumPy and Pickle Types

Training arrays use:

```python
X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int32)
pos_weight = float(...)
```

The pure Python `float` avoids Python 3.13 joblib/pickle issues with NumPy scalar types.

## XGBoost Pipeline

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
    )),
])
```

The class weight is:

```python
pos_weight = float(low_risk_count / max(elevated_risk_count, 1))
```

## Cross-Validation

The script uses `StratifiedKFold` with up to five folds. It computes ROC-AUC on validation probability scores.

The reported score is chunk-level. This means chunks from one participant can appear in both training and validation folds. A person-level held-out evaluation would be more robust and is recommended before clinical or high-stakes use.

## Training Results

The completed run produced:

```text
Unique participants processed: 103
Total 20-second voice samples: 336
Low-risk samples: 243
Elevated-risk samples: 93
Positive/elevated samples: 93
Scale pos_weight: 2.612903
StratifiedKFold ROC-AUC: 0.9368
Fold scores: 0.9248, 0.9399, 0.8991, 0.9539, 0.9660
```

## Participant IDs Used

```text
303, 304, 305, 310, 312, 313, 315, 316, 317, 318, 319, 320,
321, 322, 328, 330, 333, 336, 338, 339, 340, 341, 343, 344,
345, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360,
362, 363, 364, 366, 368, 369, 370, 371, 372, 374, 375, 376,
379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401, 402,
409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429,
430, 433, 434, 437, 441, 443, 444, 445, 446, 447, 448, 449,
454, 455, 456, 457, 459, 463, 464, 468, 471, 473, 474, 475,
478, 479, 485, 486, 487, 488, 491
```

## Participant Exclusions

Participant 314 had both:

- `314_AUDIO.wav`
- `314_TRANSCRIPT.csv`

It was excluded because no continuous PHQ-8 score was found for ID 314.

Additional exclusions:

- 79 audio participants had no merged PHQ-8 label.
- Participants 324, 325, 326, and 327 had scores but no matching extracted WAV.

The script does not fabricate missing scores or missing audio.

## Model Output

The script saves:

```text
models/saved_voice_model.pkl
```

The payload is:

```python
{
    "model": pipeline,
    "feature_names": feat_names,
}
```

The app loads this payload through [src/aria/acoustics.py](src/aria/acoustics.py).

## Retraining

From the repository root:

```powershell
python src/train_daic.py
```

The script reports:

- Unique participants processed.
- Total 20-second samples.
- Low/elevated class balance.
- Scale weight.
- Five-fold ROC-AUC.
- Saved model location.

## Compatibility Checklist

After changing training:

1. Confirm [src/aria/features.py](src/aria/features.py) still produces the same feature order.
2. Confirm the live app receives `np.float32` values.
3. Confirm `feature_names` has 39 entries.
4. Retrain before changing live inference.
5. Test model loading from `models/saved_voice_model.pkl`.
6. Run the application tests.

## Limitations

- The result is an experimental wellness screen, not a clinical diagnosis.
- Chunk-level cross-validation is not participant-level external validation.
- Voice quality varies across microphones, accents, illness, and background noise.
- The acoustic model is not evidence of depression or anxiety by itself.
- Missing DAIC labels limit participant coverage.
- XGBoost probabilities must not be interpreted as calibrated clinical probabilities.
