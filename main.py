from flask import Flask, request, jsonify
import cv2
from flask_cors import CORS
import os
import librosa
from keras.models import load_model
import shutil
import numpy as np

app = Flask(__name__)
CORS(app)

model = load_model('model_specto.h5')

def ekstrak_specto(dir, n_fft=512, hop_length=256):
    x, sr = librosa.load(dir)
    spec = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    img = cv2.resize(spec, (128, 128))

    return img

def predict_audio(model, audio_path):
    # Ekstraksi spectrogram dari audio
    spectrogram = ekstrak_specto(audio_path)
    # Normalisasi
    spectrogram = spectrogram / 255.0

    spectrogram = spectrogram.reshape(1, 128, 128, 1)
    # Prediksi menggunakan model
    prediction = model.predict(spectrogram)
    return prediction

@app.route('/prepare', methods=['GET'])
def delete_images():
    folder = 'audio'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        return f"server ready to detect {folder} again", 200
    else:
        return f"server ready", 200

@app.route('/post_audio', methods=['POST'])
def post_audio():
    file = request.files['file']
    if 'file' not in request.files:
        return "No File part", 400

    if file.filename == '':
        return "No Selected file", 400

    if file:
        if not os.path.exists('audio'):
            os.makedirs('audio')
        filename = file.filename
        aud_path = os.path.join('audio/'+filename)
        file.save(aud_path)
        return f"File {filename} uploaded successfully", 201

@app.route('/predict', methods=['GET'])
def predict():
    file = os.listdir('audio')
    if len(file) == 0:
        return "File doesnt exist", 400

    aud_path = os.path.join('audio/'+file[0])
    label = ['Hip-Hop', 'POP', 'Rock']
    prediction = predict_audio(model, aud_path)
    result = {
        "prediksi_genre": label[np.argmax(prediction)],
        "probabilitas": str("Confidence :") + str(float(prediction[0][np.argmax(prediction)]))
    }

    return jsonify(result), 200

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True, port=5000)