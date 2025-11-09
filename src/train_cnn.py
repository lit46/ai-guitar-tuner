import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = np.load("data/dataset_mels.npz", allow_pickle=True)
X, y, labels = data["X"], data["y"], data["labels"]
X = X[..., np.newaxis] / 80.0 + 1.0  # normalize dB to 0–1
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=8)
model.save("models/string_classifier.h5")
print("✅ Saved models/string_classifier.h5")
