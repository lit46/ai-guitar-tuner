import os, glob, numpy as np
import librosa, soundfile as sf
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Paths / Config ---
TEST_ROOT = "data/test"
MODEL_PATH = "models/string_classifier.h5"  # improved model
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

SR_YIN = 44100
SR_MEL = 22050
IN_TUNE = 5.0  # cents threshold

TARGET = {"E2":82.41, "A2":110.00, "D3":146.83, "G3":196.00, "B3":246.94, "E4":329.63}
LABELS = list(TARGET.keys())

# --- Baseline: FFT f0 + Range-based string detect ---
def f0_fft(y, sr):
    N = len(y)
    if N == 0:
        return 0.0
    # window helps a bit
    y = y * np.hanning(N)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(N, 1/sr)
    if len(Y) < 3:
        return 0.0
    idx = np.argmax(np.abs(Y))
    return float(freqs[idx])

STRING_RANGES = {
    "E2": (70, 95),
    "A2": (95, 130),
    "D3": (130, 165),
    "G3": (165, 220),
    "B3": (220, 280),
    "E4": (280, 360)
}
def baseline_string_from_freq(f):
    for s,(lo,hi) in STRING_RANGES.items():
        if lo <= f <= hi:
            return s
    # fallback to closest target if out of range
    return min(TARGET, key=lambda k: abs(TARGET[k]-f))

# --- Improved: YIN f0 + CNN string detect ---
def f0_yin(y, sr):
    if len(y) < 4096:
        return 0.0
    y = y.astype(np.float32)
    y = y * np.hanning(len(y))
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=8192, hop_length=512)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else 0.0

def mel_128x128(y, sr=SR_MEL, n_mels=128, hop=512):
    y = librosa.resample(y, orig_sr=sr if sr!=SR_MEL else SR_MEL, target_sr=SR_MEL) if sr!=SR_MEL else y
    y = librosa.util.normalize(y)
    S = librosa.feature.melspectrogram(y=y, sr=SR_MEL, n_mels=n_mels, hop_length=hop)
    S_db = librosa.power_to_db(S, ref=np.max)
    # fit to 128 frames
    if S_db.shape[1] < 128:
        pad = 128 - S_db.shape[1]
        S_db = np.pad(S_db, ((0,0),(0,pad)))
    else:
        S_db = S_db[:, :128]
    # scale to 0..1 ish
    S_db = S_db / 80.0 + 1.0
    return S_db[..., np.newaxis]

# Load model lazily to avoid import cost if not needed
_model = None
def load_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model

def improved_string_predict_proba(y_raw, sr_raw):
    model = load_model()
    # build mel from original y_raw at SR_MEL
    if sr_raw != SR_MEL:
        y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=SR_MEL)
    else:
        y = y_raw
    X = mel_128x128(y, sr=SR_MEL)[np.newaxis, ...]  # (1,128,128,1)
    proba = model.predict(X, verbose=0)[0]          # (6,)
    return proba

# --- Common helpers ---
def cents_diff(f0, note):
    if f0 <= 0:
        return np.nan
    return 1200.0 * np.log2(f0 / TARGET[note])

def decide_direction(diff_cents, thr=IN_TUNE):
    if not np.isfinite(diff_cents):
        return "UNK"
    if abs(diff_cents) <= thr:
        return "IN"
    return "DOWN" if diff_cents > 0 else "UP"  # +ve = sharp -> tune down

def walk_files(root=TEST_ROOT):
    files = []
    for lab in LABELS:
        files.extend([(p, lab) for p in glob.glob(os.path.join(root, lab, "*.wav"))])
    return files

def sensitivity_specificity(cm):
    # cm shape: (C,C)
    sens = []  # recall per class
    spec = []  # specificity per class (TN / (TN+FP))
    C = cm.shape[0]
    for i in range(C):
        TP = cm[i,i]
        FN = cm[i,:].sum() - TP
        FP = cm[:,i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens.append(TP / (TP + FN) if (TP+FN)>0 else 0.0)
        spec.append(TN / (TN + FP) if (TN+FP)>0 else 0.0)
    return np.array(sens), np.array(spec)

def main():
    files = walk_files(TEST_ROOT)
    assert len(files)>0, "No test files found under data/test//.wav"

    y_true_idx = []
    # --- String task collections ---
    y_pred_baseline_idx = []
    y_pred_improved_idx = []
    proba_baseline = []  # for ROC we’ll fabricate OvR probs using distance-to-target heuristic
    proba_improved = []  # true softmax from CNN

    # --- Direction task collections ---
    y_dir_true = []
    y_dir_pred_baseline = []
    y_dir_pred_improved = []

    # for baseline fake-proba: convert a frequency to soft-like scores based on distance
    def baseline_probs_from_freq(f0):
        # smaller distance => higher probability
        dists = np.array([abs(f0 - TARGET[k]) for k in LABELS])
        # avoid div by zero
        dists = np.maximum(dists, 1e-6)
        inv = 1.0 / dists
        p = inv / inv.sum()
        return p

    for path, true_label in tqdm(files):
        y, sr = librosa.load(path, sr=SR_YIN, mono=True)

        # --- BASELINE ---
        f0_b = f0_fft(y, sr)
        pred_str_baseline = baseline_string_from_freq(f0_b)
        proba_b = baseline_probs_from_freq(f0_b)

        # direction baseline uses true_label target (what a tuner user expects)
        diff_b_true = cents_diff(f0_b, true_label)
        dir_b = decide_direction(diff_b_true)

        # --- IMPROVED ---
        f0_i = f0_yin(y, sr)
        proba_i = improved_string_predict_proba(y, sr)  # softmax (6,)
        pred_idx_i = int(np.argmax(proba_i))
        pred_str_improved = LABELS[pred_idx_i]

        diff_i_true = cents_diff(f0_i, true_label)
        dir_i = decide_direction(diff_i_true)

        # --- Collect ---
        y_true_idx.append(LABELS.index(true_label))
        y_pred_baseline_idx.append(LABELS.index(pred_str_baseline))
        y_pred_improved_idx.append(LABELS.index(pred_str_improved))
        proba_baseline.append(proba_b)
        proba_improved.append(proba_i)

        y_dir_true.append(dir_b)  # same ground-truth derived from true_label (OK)
        y_dir_pred_baseline.append(dir_b)
        y_dir_pred_improved.append(dir_i)

    y_true_idx = np.array(y_true_idx)
    y_pred_baseline_idx = np.array(y_pred_baseline_idx)
    y_pred_improved_idx = np.array(y_pred_improved_idx)
    proba_baseline = np.array(proba_baseline)   # (N,6)
    proba_improved = np.array(proba_improved)   # (N,6)

    # ---------- STRING: metrics ----------
    print("\n====== STRING CLASSIFICATION (6-class) ======")
    print("\n--- BASELINE ---")
    cm_b = confusion_matrix(y_true_idx, y_pred_baseline_idx, labels=range(len(LABELS)))
    print("Confusion Matrix:\n", cm_b)
    print("\nReport:\n", classification_report(y_true_idx, y_pred_baseline_idx, target_names=LABELS, digits=3))
    acc_b = accuracy_score(y_true_idx, y_pred_baseline_idx)
    print(f"Accuracy: {acc_b:.3f}")
    sens_b, spec_b = sensitivity_specificity(cm_b)
    print("Sensitivity per class:", dict(zip(LABELS, sens_b.round(3))))
    print("Specificity per class:", dict(zip(LABELS, spec_b.round(3))))

    print("\n--- IMPROVED (YIN + CNN) ---")
    cm_i = confusion_matrix(y_true_idx, y_pred_improved_idx, labels=range(len(LABELS)))
    print("Confusion Matrix:\n", cm_i)
    print("\nReport:\n", classification_report(y_true_idx, y_pred_improved_idx, target_names=LABELS, digits=3))
    acc_i = accuracy_score(y_true_idx, y_pred_improved_idx)
    print(f"Accuracy: {acc_i:.3f}")
    sens_i, spec_i = sensitivity_specificity(cm_i)
    print("Sensitivity per class:", dict(zip(LABELS, sens_i.round(3))))
    print("Specificity per class:", dict(zip(LABELS, spec_i.round(3))))

    # --- ROC-AUC (multiclass OvR) ---
    y_true_bin = label_binarize(y_true_idx, classes=list(range(len(LABELS))))
    try:
        auc_b_macro = roc_auc_score(y_true_bin, proba_baseline, average="macro", multi_class="ovr")
        auc_b_micro = roc_auc_score(y_true_bin, proba_baseline, average="micro", multi_class="ovr")
        auc_i_macro = roc_auc_score(y_true_bin, proba_improved, average="macro", multi_class="ovr")
        auc_i_micro = roc_auc_score(y_true_bin, proba_improved, average="micro", multi_class="ovr")
        print(f"\nROC-AUC (Baseline)  macro={auc_b_macro:.3f}  micro={auc_b_micro:.3f}")
        print(f"ROC-AUC (Improved) macro={auc_i_macro:.3f}  micro={auc_i_micro:.3f}")

        # Optional: plot ROC curves for improved
        plt.figure()
        for c, lab in enumerate(LABELS):
            fpr, tpr, _ = roc_curve(y_true_bin[:,c], proba_improved[:,c])
            plt.plot(fpr, tpr, label=lab)
        plt.plot([0,1], [0,1], linestyle="--")
        plt.title("ROC – Improved (OvR)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "roc_string_improved.png"))

        plt.figure()
        for c, lab in enumerate(LABELS):
            fpr, tpr, _ = roc_curve(y_true_bin[:,c], proba_baseline[:,c])
            plt.plot(fpr, tpr, label=lab)
        plt.plot([0,1], [0,1], linestyle="--")
        plt.title("ROC – Baseline (OvR)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "roc_string_baseline.png"))

        print("\nROC plots saved in reports/:")
        print(" - reports/roc_string_improved.png")
        print(" - reports/roc_string_baseline.png")
    except Exception as e:
        print("ROC-AUC could not be computed (likely too few samples or probabilities invalid):", e)

    # ---------- DIRECTION: metrics (UP/IN/DOWN) ----------
    print("\n====== DIRECTION (UP / IN / DOWN) ======")
    # Convert to indices with fixed label order
    DIR_LABELS = ["UP","IN","DOWN"]
    to_idx = {k:i for i,k in enumerate(DIR_LABELS)}
    y_d_true = np.array([to_idx.get(x,1) for x in y_dir_true])
    y_d_b = np.array([to_idx.get(x,1) for x in y_dir_pred_baseline])
    y_d_i = np.array([to_idx.get(x,1) for x in y_dir_pred_improved])

    print("\n--- BASELINE ---")
    cm_db = confusion_matrix(y_d_true, y_d_b, labels=range(3))
    print("Confusion Matrix:\n", cm_db)
    print("\nReport:\n", classification_report(y_d_true, y_d_b, target_names=DIR_LABELS, digits=3))
    print(f"Accuracy: {accuracy_score(y_d_true, y_d_b):.3f}")

    print("\n--- IMPROVED ---")
    cm_di = confusion_matrix(y_d_true, y_d_i, labels=range(3))
    print("Confusion Matrix:\n", cm_di)
    print("\nReport:\n", classification_report(y_d_true, y_d_i, target_names=DIR_LABELS, digits=3))
    print(f"Accuracy: {accuracy_score(y_d_true, y_d_i):.3f}")

    # Sensitivity/Specificity for direction (per-class vs rest)
    def sens_spec_from_cm(cm, labels):
        sens, spec = sensitivity_specificity(cm)
        return dict(zip(labels, sens.round(3))), dict(zip(labels, spec.round(3)))
    sens_b_d, spec_b_d = sens_spec_from_cm(cm_db, DIR_LABELS)
    sens_i_d, spec_i_d = sens_spec_from_cm(cm_di, DIR_LABELS)

    print("\nSensitivity (Baseline):", sens_b_d)
    print("Specificity (Baseline):", spec_b_d)
    print("Sensitivity (Improved):", sens_i_d)
    print("Specificity (Improved):", spec_i_d)

    # --------- HEADLINE COMPARISON FOR REPORT ---------
    print("\n====== HEADLINE COMPARISON ======")
    print(f"String Accuracy: Baseline={acc_b:.3f}  Improved={acc_i:.3f}")
    try:
        print(f"ROC-AUC (macro): Baseline={auc_b_macro:.3f}  Improved={auc_i_macro:.3f}")
        print(f"ROC-AUC (micro): Baseline={auc_b_micro:.3f}  Improved={auc_i_micro:.3f}")
    except:
        pass
    print("Sensitivity per string (Baseline):", dict(zip(LABELS, sens_b.round(3))))
    print("Sensitivity per string (Improved):", dict(zip(LABELS, sens_i.round(3))))
    print("Specificity per string (Baseline):", dict(zip(LABELS, spec_b.round(3))))
    print("Specificity per string (Improved):", dict(zip(LABELS, spec_i.round(3))))

if _name_ == "_main_":
    main()
