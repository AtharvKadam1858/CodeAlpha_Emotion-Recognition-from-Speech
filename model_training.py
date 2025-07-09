import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
import joblib

DATA_PATH = "ravdess_data"

emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file_path, max_pad_len=862):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

X, Y = [], []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            print(f"Processing file: {file}")  # Debug print
            parts = file.split("-")
            if len(parts) > 2:
                emotion_code = parts[2]
                print(f"Emotion code extracted: {emotion_code}")  # Debug print
                if emotion_code in emotion_labels:
                    emotion = emotion_labels[emotion_code]
                    file_path = os.path.join(root, file)
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        Y.append(emotion)
                else:
                    print(f"Unknown emotion code {emotion_code} in file {file}")
            else:
                print(f"Skipping file due to unexpected filename format: {file}")

print(f"Total samples loaded: {len(X)}")
print(f"Total labels loaded: {len(Y)}")

if len(X) == 0 or len(Y) == 0:
    raise ValueError("No data loaded. Check your dataset path and filenames.")

X = np.array(X)
Y = np.array(Y)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

if len(Y_encoded) == 0:
    raise ValueError("No labels found after encoding. Check your labels.")

Y_onehot = to_categorical(Y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 40, -1)
X_test = X_test.reshape(X_test.shape[0], 40, -1)

model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(40, X_train.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(Y_onehot.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.2f}%")

model.save('saved_model.h5')
joblib.dump(le, 'label_encoder.pkl')
