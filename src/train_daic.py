import os
import re
import zipfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb

from Front-End.features import extract_features

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def prepare_daic_folders(data_dir="daic_data"):
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    fallback_path = PROJECT_ROOT / "data" / "daic_data"
    if not data_path.exists() and fallback_path.exists():
        data_path = fallback_path

    data_path.mkdir(parents=True, exist_ok=True)

    unused_patterns = (
        "*CLNF*",
        "*.bin",
        "*.mp4",
        "*.avi",
        "*FORMANT*",
        "*COVAREP*",
    )
    removed_count = 0
    for pattern in unused_patterns:
        for file_path in data_path.rglob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    removed_count += 1
                except OSError as exc:
                    print(f"Could not remove {file_path}: {exc}")

    extracted_count = 0
    for zip_path in sorted(data_path.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue

                    member_name = member.filename.replace("\\", "/")
                    base_name = Path(member_name).name
                    if not (
                        base_name.endswith("_AUDIO.wav")
                        or base_name.endswith("_TRANSCRIPT.csv")
                    ):
                        continue

                    pid_match = re.match(r"^(\d+)_", base_name)
                    if not pid_match:
                        continue

                    participant_dir = data_path / pid_match.group(1)
                    participant_dir.mkdir(parents=True, exist_ok=True)
                    target_path = participant_dir / base_name

                    if target_path.exists():
                        continue

                    with archive.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())
                    extracted_count += 1
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"Could not process archive {zip_path}: {exc}")

    if removed_count:
        print(
            f"Disk cleanup removed {removed_count} unused video/feature file(s)."
        )
    if extracted_count:
        print(
            f"Selective extraction added {extracted_count} participant audio/transcript file(s)."
        )

    return data_path


def load_all_score_maps(data_dir):
    data_path = Path(data_dir)
    score_map = {}

    for csv_path in sorted(data_path.rglob("*.csv")):
        if csv_path.name.upper().endswith("_TRANSCRIPT.CSV"):
            continue

        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Skipping unreadable score CSV {csv_path}: {exc}")
            continue

        columns = list(frame.columns)
        id_columns = [
            column for column in columns if "id" in str(column).strip().lower()
        ]
        score_columns = [
            column
            for column in columns
            if "phq" in str(column).strip().lower()
            and "binary" not in str(column).strip().lower()
        ]

        if not id_columns or not score_columns:
            continue

        id_col = id_columns[0]
        score_col = score_columns[0]

        for _, row in frame.iterrows():
            try:
                participant_id = int(float(row[id_col]))
                phq8_score = float(row[score_col])
                if np.isfinite(phq8_score):
                    score_map[participant_id] = phq8_score
            except (TypeError, ValueError, OverflowError):
                continue

    return score_map


def get_participant_audio_chunks(
    wav_path,
    transcript_path,
    target_sr=16000,
    chunk_sec=20,
    max_chunks=3,
):
    transcript_path = Path(transcript_path)

    try:
        frame = pd.read_csv(transcript_path, sep="\t")
        normalized_columns = [str(column).strip().lower() for column in frame.columns]
        if "speaker" not in normalized_columns:
            frame = pd.read_csv(transcript_path, sep=",")
    except Exception:
        frame = pd.read_csv(transcript_path, sep=",")

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required_columns = {"speaker", "start_time", "stop_time"}
    if not required_columns.issubset(set(frame.columns)):
        raise ValueError("Transcript columns must include speaker, start_time, and stop_time.")

    participant_rows = frame[
        frame["speaker"].astype(str).str.strip().str.lower() == "participant"
    ].copy()
    if participant_rows.empty:
        raise ValueError("No Participant rows found.")

    if len(participant_rows) > 4:
        participant_rows = participant_rows.iloc[2:]

    full_audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    clips = []

    for _, row in participant_rows.iterrows():
        try:
            start_seconds = float(row["start_time"])
            stop_seconds = float(row["stop_time"])
        except (TypeError, ValueError):
            continue

        start_sample = int(start_seconds * sr)
        stop_sample = int(stop_seconds * sr)
        if stop_sample > start_sample and start_sample >= 0 and stop_sample <= len(full_audio):
            clips.append(full_audio[start_sample:stop_sample])

    if not clips:
        raise ValueError("Transcript contained no usable participant audio segments.")

    clean_audio = np.concatenate(clips)
    peak = float(np.max(np.abs(clean_audio)))
    if peak > 0:
        clean_audio = clean_audio / peak

    chunk_samples = int(chunk_sec * target_sr)
    minimum_chunk_samples = int(8 * target_sr)
    chunks = []

    for chunk_index in range(max_chunks):
        start = chunk_index * chunk_samples
        if start >= len(clean_audio):
            break

        chunk = clean_audio[start:start + chunk_samples]
        if len(chunk) < minimum_chunk_samples:
            if chunk_index == 0:
                continue
            break

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append((chunk, target_sr))

    if not chunks:
        raise ValueError("No participant chunk reached at least 8 seconds.")

    return chunks


def main():
    data_dir = prepare_daic_folders("daic_data")
    score_map = load_all_score_maps(data_dir)

    if not score_map:
        print(
            "ERROR: No participant PHQ-8 scores were found in DAIC CSV files "
            f"under {data_dir}."
        )
        return

    X_list = []
    y_list = []
    participant_ids_processed = set()

    for wav_path in sorted(data_dir.rglob("*_AUDIO.wav")):
        pid_match = re.match(r"^(\d+)_AUDIO\.wav$", wav_path.name, re.IGNORECASE)
        if not pid_match:
            continue

        participant_id = int(pid_match.group(1))
        if participant_id not in score_map:
            continue

        transcript_path = wav_path.with_name(
            f"{participant_id}_TRANSCRIPT.csv"
        )
        if not transcript_path.exists():
            continue

        try:
            chunks = get_participant_audio_chunks(
                wav_path,
                transcript_path,
                target_sr=16000,
                chunk_sec=20,
                max_chunks=3,
            )

            participant_processed = False
            for chunk, sample_rate in chunks:
                vec, feat_names, _ = extract_features(chunk, sample_rate)
                X_list.append(vec)
                y_list.append(
                    1 if score_map[participant_id] >= 10 else 0
                )
                participant_processed = True

            if participant_processed:
                participant_ids_processed.add(participant_id)
                print(
                    f"Processed Participant {participant_id}: "
                    f"{len(chunks)} chunk(s), "
                    f"PHQ-8={score_map[participant_id]:g}"
                )
        except Exception as exc:
            print(f"Skipping Participant {participant_id}: {exc}")

    if len(X_list) < 4:
        print(
            "WARNING: Fewer than 4 voice samples were processed. "
            "Check that participant ZIP files, audio, and transcripts are present "
            f"inside {data_dir}."
        )
        return

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    low_risk_count = int(np.sum(y == 0))
    elevated_risk_count = int(np.sum(y == 1))

    positive_count = elevated_risk_count
    pos_weight = float(
        (y == 0).sum() / max((y == 1).sum(), 1)
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=pos_weight,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipeline.fit(X, y)

    model_payload = {"model": pipeline, "feature_names": feat_names}
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "saved_voice_model.pkl"

    joblib.dump(model_payload, model_path, compress=3)

    print("\n========== Training Summary ==========")
    print(f"Unique participants processed: {len(participant_ids_processed)}")
    print(f"Total 20-second voice samples: {len(X)}")
    print(
        f"Class balance: Low risk={low_risk_count}, "
        f"Elevated risk={elevated_risk_count}"
    )
    print(f"Positive class: {positive_count}")
    print(f"Scale pos_weight: {pos_weight:.6f}")

    class_counts = np.bincount(y, minlength=2)
    if len(np.unique(y)) == 2 and np.min(class_counts) >= 2:
        n_splits = min(5, int(np.min(class_counts)))
        cross_validator = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        fold_scores = []

        for train_indices, validation_indices in cross_validator.split(X, y):
            fold_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        xgb.XGBClassifier(
                            n_estimators=150,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            scale_pos_weight=pos_weight,
                            random_state=42,
                            eval_metric="logloss",
                        ),
                    ),
                ]
            )
            fold_pipeline.fit(X[train_indices], y[train_indices])
            validation_probabilities = fold_pipeline.predict_proba(
                X[validation_indices]
            )[:, 1]
            fold_scores.append(
                float(
                    roc_auc_score(
                        y[validation_indices],
                        validation_probabilities,
                    )
                )
            )

        print(
            f"StratifiedKFold ROC-AUC ({n_splits} folds): "
            f"{np.mean(fold_scores):.4f} "
            f"(fold scores: {[round(score, 4) for score in fold_scores]})"
        )
    else:
        print("StratifiedKFold ROC-AUC: unavailable (need at least 2 samples per class)")

    print(f"\nSUCCESS: saved trained model to {model_path}")


if __name__ == "__main__":
    main()
