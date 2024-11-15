import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras  
import json
import noisereduce as nr
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")


try:
    with open('config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Ошибка: Файл config.json не найден.")
    exit(1)
except json.JSONDecodeError:
    print("Ошибка: Не удалось декодировать config.json. Проверьте синтаксис файла.")
    exit(1)

# Путь к модели (из config.json)
model_path = os.path.join(config['audio_model_path'])
model = keras.models.load_model(model_path)


# Функция для оценки уровня шума
def estimate_noise_level(y):
    return np.mean(librosa.feature.rms(y=y)) 


def reduce_noise(y, sr, noise_level_threshold=0.05):
    """Применяет шумоподавление, если уровень шума превышает порог."""
    try:
        noise_level = estimate_noise_level(y)

        if noise_level > noise_level_threshold:
            reduced_noise = nr.reduce_noise(y=y, sr=sr) 
            return reduced_noise
        else:
            return y
    except Exception as e:
        print(f"Ошибка при шумоподавлении: {e}")
        return y  


def extract_audio_features(filename):
    try:
        # Конвертируем аудиофайл в WAV
        if not filename.endswith(".wav"):
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            data, samplerate = sf.read(filename)
            sf.write(wav_filename, data, samplerate)
            filename = wav_filename  

        audio_data, sr = librosa.load(filename)

        # Шумоподавление
        #audio_data = reduce_noise(audio_data, sr) 

        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=audio_data, sr=sr).T, axis=0)

        print("Фичи для аудио извлечены.")

        # Удаляем временный файл
        os.remove(filename)
        
        return mfccs, chroma, mel, contrast, tonnetz

    except Exception as e:
        print(f"Ошибка при извлечении признаков: {e}")
        return None, None, None, None, None 



def predict_emotion_from_audio(mfccs, chroma, mel, contrast, tonnetz):
    try:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz]) 

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.reshape(1, -1))
        scaled_features  = tf.reshape(scaled_features, [1, 193, 1])

        prediction = model.predict(scaled_features)
        predicted_emotion_index = np.argmax(prediction)

        emotions = {
            0: "neutral",
            1: "anger",
            2: "enthusiasm",
            3: "fear",
            4: "sadness",
            5: "happiness",
            6: "disgust"
        }

        predicted_emotion = emotions.get(predicted_emotion_index, "unknown")

        print("Эмоция из аудио предсказана.")
        return predicted_emotion

    except Exception as e:
        print(f"Ошибка при распознавании эмоции из аудио: {e}")
        return None