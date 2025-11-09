import sounddevice as sd
import numpy as np
import scipy.fftpack as fft
import time
import os
from colorama import Fore, Style, init
import librosa

init(autoreset=True)

SR = 44100
DURATION = 0.5  # smaller chunk for smoother updates

NOTE_FREQS = {
    "E2": 82.41, "A2": 110.00, "D3": 146.83,
    "G3": 196.00, "B3": 246.94, "E4": 329.63
}

def closest_note(freq):
    if freq <= 0:
        return None, 0
    note = min(NOTE_FREQS, key=lambda n: abs(NOTE_FREQS[n] - freq))
    diff = 1200 * np.log2(freq / NOTE_FREQS[note])  # difference in cents
    return note, diff

def get_freq(data, rate):
    # librosa expects float32 mono array
    y = np.array(data, dtype=np.float32).flatten()
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=rate)
    # Take median over valid frames to ignore transients
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) > 0 else 0

def display_tuner(note, diff):
    os.system('cls' if os.name == 'nt' else 'clear')

    # Define visual scale
    scale = np.linspace(-50, 50, 41)
    pos = int((diff + 50) / 100 * 40)
    pos = max(0, min(40, pos))

    line = ['-'] * 41
    line[20] = '|'
    line[pos] = '🔻'

    color = Fore.GREEN if abs(diff) < 3 else Fore.RED

    print(f"{color}{note} |{''.join(line)}| ({diff:+.2f} cents)")
    if abs(diff) < 3:
        print(Fore.GREEN + "✓ In Tune")
    elif diff > 0:
        print(Fore.RED + "Too Sharp (Tune Down)")
    else:
        print(Fore.RED + "Too Flat (Tune Up)")

def run_tuner():
    print("🎸 Real-Time Tuner (Ctrl+C to stop)\n")
    time.sleep(1)
    while True:
        data = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        freq = get_freq(data[:, 0], SR)
        note, diff = closest_note(freq)
        if note:
            display_tuner(note, diff)
        else:
            print("Listening...")

if __name__ == "__main__":
    run_tuner()
