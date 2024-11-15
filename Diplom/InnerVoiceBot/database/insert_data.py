import json
import psycopg2
import numpy as np


with open('config.json', 'r') as config_file:
    config = json.load(config_file)

database_creds_file = config['database_creds_file']

with open(database_creds_file, 'r') as database_creds_file:
    database_creds = json.load(database_creds_file)

try:
    conn = psycopg2.connect(**database_creds)
    cursor = conn.cursor()
    print("Успешно подключились к базе данных")
except Exception as e:
    print(f"Ошибка при подключении к базе данных: {e}")
    exit(1)  


# Функция для сохранения информации о голосовом сообщении в базе данных
def save_audio_message_info_to_db(filename, user_id, message_id, emotion, mfccs, chroma, mel, contrast, tonnetz):
    try:
        # Преобразование массива в строку
        mfccs_str = np.array2string(mfccs)
        chroma_str = np.array2string(chroma)
        mel_str = np.array2string(mel)
        contrast_str = np.array2string(contrast)
        tonnetz_str = np.array2string(tonnetz)

        cursor.execute("""
            INSERT INTO audio_features (message_id, mfccs, chroma, mel, contrast, tonnetz)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (message_id, mfccs_str, chroma_str, mel_str, contrast_str, tonnetz_str))

        cursor.execute("""
            INSERT INTO audio_messages (filename, user_id, message_id, emotion)
            VALUES (%s, %s, %s, %s)
        """, (filename, user_id, message_id, emotion))

        conn.commit()
        print("Информация о голосовом сообщении сохранена в базе данных.")
    except Exception as e:
        print(f"Ошибка при сохранении информации о голосовом сообщении в базе данных: {e}")


# Функция для сохранения информации о текстовом сообщении в базе данных
def save_text_message_info_to_db(user_id, message_id, text, processed_text, emotion):
    try:
        cursor.execute("""
            INSERT INTO text_messages (user_id, message_id, text, processed_text, emotion) 
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, message_id, text, processed_text, emotion))
        conn.commit()
        print("Информация о текстовом сообщении сохранена в базе данных.")
    except Exception as e:
        print(f"Ошибка при сохранении информации о текстовом сообщении в базе данных: {e}")