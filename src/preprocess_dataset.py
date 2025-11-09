import librosa, numpy as np, os, glob
from tqdm import tqdm

SR = 22050
N_MELS = 128
HOP = 512

def process_wav(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = librosa.util.normalize(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
    S_db = librosa.power_to_db(S, ref=np.max)
    # resize to fixed length (128×128)
    if S_db.shape[1] < 128:
        pad = 128 - S_db.shape[1]
        S_db = np.pad(S_db, ((0,0),(0,pad)))
    else:
        S_db = S_db[:, :128]
    return S_db

def build_dataset(data_root="data/train"):
    X, y, labels = [], [], sorted(os.listdir(data_root))
    label_map = {lbl:i for i,lbl in enumerate(labels)}
    for label in labels:
        for wav_path in tqdm(glob.glob(os.path.join(data_root, label, "*.wav"))):
            mel = process_wav(wav_path)
            X.append(mel)
            y.append(label_map[label])
    X, y = np.array(X), np.array(y)
    np.savez("data/dataset_mels.npz", X=X, y=y, labels=labels)
    print(f"Saved dataset_mels.npz with shape {X.shape}")
    return label_map

if __name__ == "__main__":
    build_dataset()
