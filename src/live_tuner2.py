import sounddevice as sd
import numpy as np
import scipy.fftpack as fft
import tkinter as tk
from tkinter import Canvas
import threading
import time

SR = 44100
DURATION = 0.5

NOTE_FREQS = {
    "E2": 82.41, "A2": 110.00, "D3": 146.83,
    "G3": 196.00, "B3": 246.94, "E4": 329.63
}

def closest_note(freq):
    if freq <= 0:
        return None, 0
    note = min(NOTE_FREQS, key=lambda n: abs(NOTE_FREQS[n] - freq))
    diff = 1200 * np.log2(freq / NOTE_FREQS[note])  # cents difference
    return note, diff

def get_freq(data, rate):
    N = len(data)
    yf = fft.fft(data)
    xf = np.linspace(0, rate, N)
    idx = np.argmax(np.abs(yf[:N // 2]))
    return xf[idx]

# ----------------- GUI -----------------
class TunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Guitar Tuner 🎸")
        self.root.configure(bg="#1b1b1b")

        self.note_label = tk.Label(root, text="Listening...", font=("Arial", 28, "bold"), fg="white", bg="#1b1b1b")
        self.note_label.pack(pady=20)

        self.canvas = Canvas(root, width=400, height=200, bg="#222222", highlightthickness=0)
        self.canvas.pack()

        # Draw base scale line
        self.center_x = 200
        self.canvas.create_line(50, 150, 350, 150, fill="gray", width=2)
        self.needle = self.canvas.create_line(self.center_x, 150, self.center_x, 70, width=6, fill="green")

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        threading.Thread(target=self.listen_audio, daemon=True).start()

    def close(self):
        self.running = False
        self.root.destroy()

    def listen_audio(self):
        while self.running:
            try:
                data = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype="float32")
                sd.wait()
                freq = get_freq(data[:, 0], SR)
                note, diff = closest_note(freq)
                self.root.after(0, lambda: self.update_ui(note, diff))
            except Exception as e:
                print("Error:", e)
            time.sleep(0.05)

    def update_ui(self, note, diff):
        if not note:
            self.note_label.config(text="Listening...", fg="white")
            return

        # Needle movement range: -50 cents (left) to +50 (right)
        max_angle = 80  # degrees range
        angle = max(-max_angle, min(max_angle, diff))
        x_offset = (angle / max_angle) * 150

        color = "green" if abs(diff) < 3 else "red"
        self.note_label.config(text=f"{note} ({diff:+.1f} cents)", fg=color)

        # Move needle
        self.canvas.coords(self.needle, self.center_x, 150, self.center_x + x_offset, 70)
        self.canvas.itemconfig(self.needle, fill=color)

def main():
    root = tk.Tk()
    app = TunerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
