import os
import glob
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings

from aria.features import extract_features

warnings.filterwarnings('ignore')

def get_clean_participant_audio(wav_path, transcript_path, target_sr=16000, duration=30):
    # Read transcript (handles both tab and comma separated DAIC files)
    df = pd.read_csv(transcript_path, sep='\t')
    if 'speaker' not in [c.strip().lower() for c in df.columns]:
        df = pd.read_csv(transcript_path)

    df.columns = [c.strip().lower() for c in df.columns]
    part_rows = df[df['speaker'].astype(str).str.strip().str.lower() == 'participant']

    if len(part_rows) == 0:
        raise ValueError("No Participant rows found.")

    full_audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    target_samples = target_sr * duration

    clips = []
    collected = 0
    rows_to_use = part_rows.iloc[2:] if len(part_rows) > 4 else part_rows

    for _, row in rows_to_use.iterrows():
        start_s = int(float(row['start_time']) * sr)
        stop_s = int(float(row['stop_time']) * sr)
        if stop_s > start_s and stop_s <= len(full_audio):
            clip = full_audio[start_s:stop_s]
            clips.append(clip)
            collected += len(clip)
            if collected >= target_samples:
                break

    clean_audio = np.concatenate(clips)
    if np.max(np.abs(clean_audio)) > 0:
        clean_audio = clean_audio / np.max(np.abs(clean_audio))

    if len(clean_audio) > target_samples:
        clean_audio = clean_audio[:target_samples]
    else:
        clean_audio = np.pad(clean_audio, (0, target_samples - len(clean_audio)))

    return clean_audio, sr

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data", "daic_data")
    labels_path = os.path.join(data_dir, "labels.csv")

    # 1. AUTO-CLEANER: Delete unused heavy video/3D files (_CLNF_) to free up gigabytes of disk space!
    unused_files = glob.glob(os.path.join(data_dir, "**", "*CLNF*"), recursive=True)
    for f in unused_files:
        try:
            os.remove(f)
        except Exception:
            pass
    if unused_files:
        print(f"🧹 Freed up disk space by removing {len(unused_files)} unused video tracking files!")

    if not os.path.exists(labels_path):
        print(f"ERROR: Could not find {labels_path}. Make sure you renamed your score CSV to labels.csv!")
        return

    labels_df = pd.read_csv(labels_path)
    id_col = [c for c in labels_df.columns if 'id' in c.lower()][0]
    score_col = [c for c in labels_df.columns if 'phq' in c.lower() and 'binary' not in c.lower()][0]

    X_list, y_list = [], []

    for _, row in labels_df.iterrows():
        pid = int(row[id_col])
        score = float(row[score_col])

        wav_files = glob.glob(os.path.join(data_dir, "**", f"{pid}_AUDIO.wav"), recursive=True)
        csv_files = glob.glob(os.path.join(data_dir, "**", f"{pid}_TRANSCRIPT.csv"), recursive=True)

        if not wav_files or not csv_files:
            continue

        try:
            print(f"Processing Participant {pid} (PHQ-8 Score: {score})...")
            audio, sr = get_clean_participant_audio(wav_files[0], csv_files[0])
            vec, feat_names, _ = extract_features(audio, sr)
            X_list.append(vec)
            y_list.append(1 if score >= 10 else 0)
        except Exception as e:
            print(f"  Skipping {pid}: {e}")

    if len(X_list) < 4:
        print("WARNING: Fewer than 4 participants processed. Check that your folders are unzipped inside daic_data/!")
        return

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"\nExtracted {len(X)} participants! (Low Risk: {np.sum(y==0)}, Elevated Risk: {np.sum(y==1)})")

    # Convert to pure Python float so Python 3.13 pickle/joblib doesn't bloat memory
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss'
        ))
    ])

    pipeline.fit(X, y)
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "saved_voice_model.pkl")
    joblib.dump({'model': pipeline, 'feature_names': feat_names}, model_path, compress=3)
    print(f"\n✅ SUCCESS! '{model_path}' is built and ready for the live app!")

if __name__ == "__main__":
    main()