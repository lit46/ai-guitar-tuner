# src/gui_tuner.py
import sounddevice as sd
import numpy as np
import scipy.fftpack as fft
import tkinter as tk
from tkinter import Canvas
import threading
import time
import collections
import librosa
from tensorflow.keras.models import load_model

SR = 44100
DURATION = 0.5  # seconds of recording
FMIN = 70.0
FMAX = 400.0

# Standard guitar string frequencies (Hz)
NOTE_FREQS = {
    "E2": 82.41, "A2": 110.00, "D3": 146.83,
    "G3": 196.00, "B3": 246.94, "E4": 329.63
}

# Define rough ranges for automatic string detection
STRING_RANGES = {
    "E2": (70, 95),
    "A2": (95, 130),
    "D3": (130, 165),
    "G3": (165, 220),
    "B3": (220, 280),
    "E4": (280, 360)
}

# smoothing buffers
freq_history = collections.deque(maxlen=5)   # median smoothing on f0
needle_smooth = 0.0                          # EMA for needle x_offset
NEEDLE_ALPHA = 0.25                          # smoothing factor for needle easing

def closest_note(freq):
    if freq <= 0:
        return None, 0.0
    note = min(NOTE_FREQS, key=lambda n: abs(NOTE_FREQS[n] - freq))
    diff = 1200.0 * np.log2(float(freq) / NOTE_FREQS[note])  # cents difference
    return note, diff

model = load_model("models/string_classifier.h5")
LABELS = ["E2","A2","D3","G3","B3","E4"]

def ai_detect_string(y, sr=22050):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < 128:
        pad = 128 - S_db.shape[1]
        S_db = np.pad(S_db, ((0,0),(0,pad)))
    else:
        S_db = S_db[:, :128]
    S_db = S_db[np.newaxis, ..., np.newaxis]
    S_db = S_db / 80.0 + 1.0
    pred = model.predict(S_db)
    return LABELS[int(np.argmax(pred))]

def get_freq(y, rate):
    """
    Robust f0 estimate using librosa.yin across the recorded buffer.
    Returns median of valid f0 frames or 0 on failure.
    """
    try:
        # librosa.yin expects 1D float array
        y = np.asarray(y, dtype='float32').flatten()
        if len(y) < 2048:
            return 0.0
        # apply a mild window to reduce edge effects
        y = y * np.hanning(len(y))
        # choose a frame_length that is reasonable (power of two) but < len(y)
        frame_length = 2048 if len(y) >= 2048 else len(y)
        hop_length = 512
        f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=rate,
                         frame_length=frame_length, hop_length=hop_length)
        f0_valid = f0[np.isfinite(f0)]
        if f0_valid.size == 0:
            return 0.0
        # median gives robustness to outliers/transients
        return float(np.median(f0_valid))
    except Exception as e:
        # if librosa fails for any reason, return 0
        # print("YIN error:", e)
        return 0.0

# ---------------- GUI -----------------
class TunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Guitar Tuner 🎸")
        self.root.configure(bg="#101010")

        self.note_label = tk.Label(root, text="Listening...", font=("Arial", 30, "bold"), fg="white", bg="#101010")
        self.note_label.pack(pady=10)

        self.string_label = tk.Label(root, text="", font=("Arial", 20, "italic"), fg="gray", bg="#101010")
        self.string_label.pack(pady=5)

        self.canvas = Canvas(root, width=480, height=220, bg="#222222", highlightthickness=0)
        self.canvas.pack(pady=10)

        # draw scale & center marker
        self.center_x = 240
        self.canvas.create_line(40, 160, 440, 160, fill="gray", width=3)
        self.canvas.create_text(self.center_x, 170, text="0¢", fill="white", font=("Arial", 10))
        # draw green in-tune band
        self.canvas.create_rectangle(self.center_x - 30, 145, self.center_x + 30, 155, fill="#2ecc71", outline="")
        # initial needle
        self.needle = self.canvas.create_line(self.center_x, 160, self.center_x, 60, width=6, fill="green", capstyle="round")

        # labels for cents
        self.canvas.create_text(40, 130, text="-50¢", fill="white", font=("Arial", 10))
        self.canvas.create_text(440, 130, text="+50¢", fill="white", font=("Arial", 10))

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self.listen_audio, daemon=True).start()

    def close(self):
        self.running = False
        self.root.destroy()

    def listen_audio(self):
        # continuously record short buffers and update UI
        while self.running:
            try:
                data = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype="float32")
                sd.wait()
                freq = get_freq(data[:, 0], SR)

                # smoothing: median filter over recent f0s
                if freq > 0:
                    freq_history.append(freq)
                    if len(freq_history) >= 1:
                        smoothed_freq = float(np.median(np.array(freq_history)))
                    else:
                        smoothed_freq = freq
                else:
                    smoothed_freq = 0.0

                note, diff = closest_note(smoothed_freq) if smoothed_freq > 0 else (None, 0.0)
                y = np.array(data[:,0], dtype=np.float32)
                try:
                    string = ai_detect_string(y, SR)
                except Exception:
                    string = None
                # schedule UI update on main thread
                self.root.after(0, lambda n=note, d=diff, f=smoothed_freq, s=string: self.update_ui(n, d, f, s))
            except Exception as e:
                # print("listen_audio error:", e)
                pass
            time.sleep(0.01)

    def update_ui(self, note, diff, freq, string):
        # update textual labels
        if not note or freq <= 0:
            self.note_label.config(text="Listening...", fg="white")
            self.string_label.config(text="")
            return

        color = "green" if abs(diff) < 3 else "red"
        self.note_label.config(text=f"{note}  {freq:.1f} Hz  ({diff:+.1f}¢)", fg=color)
        if string:
            self.string_label.config(text=f"Detected string: {string}", fg="#00bfff")
        else:
            self.string_label.config(text="Detected string: Unknown", fg="gray")

        # needle: map diff in cents to x-offset between -150 .. +150 for ±80 cents clamp
        max_angle = 80.0
        clamped = max(-max_angle, min(max_angle, diff))
        x_offset_target = (clamped / max_angle) * 150.0

        # easing for smooth needle motion (EMA)
        global needle_smooth
        needle_smooth = NEEDLE_ALPHA * x_offset_target + (1.0 - NEEDLE_ALPHA) * needle_smooth

        # update needle position
        x_end = self.center_x + needle_smooth
        self.canvas.coords(self.needle, self.center_x, 160, x_end, 60)
        self.canvas.itemconfig(self.needle, fill=color)

def main():
    root = tk.Tk()
    app = TunerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
