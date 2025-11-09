import sounddevice as sd
import soundfile as sf
import os
import time

SR = 44100
DURATION = 2.0  # seconds per clip
STRINGS = ["E2", "A2", "D3", "G3", "B3", "E4"]

def record_clip(label, index, out_dir="data/train"):
    os.makedirs(os.path.join(out_dir, label), exist_ok=True)
    path = os.path.join(out_dir, label, f"{label}_{index:02d}.wav")

    print(f"🎸 Recording {label}_{index:02d}.wav...")
    data = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, data, SR)
    print(f"✅ Saved: {path}\n")

def main():
    print("🎙️ AI Guitar Tuner Dataset Recorder")
    for string in STRINGS:
        input(f"\nPrepare to pluck {string}. Press Enter to start recording 5 clips.")
        for i in range(1, 6):
            record_clip(string, i)
            time.sleep(1.0)
    print("🎯 All recordings complete!")

if __name__ == "__main__":
    main()
