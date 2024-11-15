import telebot
import json
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_scripts.text_model import preprocess_text, predict_emotion_from_text
from models_scripts.audio_model import extract_audio_features, predict_emotion_from_audio
from database.insert_data import save_audio_message_info_to_db, save_text_message_info_to_db



with open('config.json', 'r') as config_file:
    config = json.load(config_file)

telegram_creds_file = config['telegram_creds_file']

with open(telegram_creds_file, 'r') as telegram_creds_file:
    telegram_creds = json.load(telegram_creds_file)
    bot_token = telegram_creds['telegram_bot_token']

# Создание экземпляра бота Telegram
bot = telebot.TeleBot(bot_token)



# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        bot.reply_to(message, '''
👋 Я бот-эмодзи InnerVoice!
Готов поделиться своими эмоциями в ответ на твои сообщения. 
Общайся со мной текстом 💬 или голосом 🎤, и я покажу тебе, что чувствую! 🤩
''')
    except Exception as e:
        print(f"Ошибка при отправке приветственного сообщения: {e}")


emojis = {
       "neutral": ["😐", "😶", "🤔", "😌"],
       "happpiness": ["😊", "😁", "😂", "🤣", "😃", "😄", "😆", "🥰", "🥳", "🤩"],
       "sadness": ["😔", "🙁", "😞", "😟", "😢", "😥", "😭", "😓", "💔", "🥺"],
       "anger": ["😡", "😠", "🤬", "😤", "👿", "😾", "💢", "💥"],
       "fear": ["😨", "😱", "😰", "😥", "😓", "🥶", "😳", "😬"],
       "disgust": ["🤢", "🤮", "🤧", "😷", "🥴", "😒", "😖", "😣"],
       "surprise": ["😮", "😯", "😲", "😳", "🤯", "🤩", "👀", "❗", "❕"],
       "enthusiasm": ["🤩", "🥳", "😁", "😃", "😄", "😆", "🤗","🔥", "💪", "💯", "🙌", "🚀","✨", "🎉"]
   }

def get_emoji_for_emotion(emotion):
    # Emotion mappings
    emotion_mapping = {
        "joy": "happpiness"
    }
    emotion = emotion_mapping.get(emotion, emotion)  

    if emotion in emojis:
        return random.choice(emojis[emotion])  
    else:
        return "😐"


# Обработчик голосовых сообщений
@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    try:
        file_id = message.voice.file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path

        downloaded_file = bot.download_file(file_path)

        filename = f"voice_message_{message.message_id}.ogg"  
        with open(filename, 'wb') as new_file:
            new_file.write(downloaded_file)

        mfccs, chroma, mel, contrast, tonnetz = extract_audio_features(filename)
        emotion = predict_emotion_from_audio(mfccs, chroma, mel, contrast, tonnetz)

        emoji = get_emoji_for_emotion(emotion)
        
        bot.reply_to(message, f"{emoji}")  
        print("Ответ пользователю отправлен.")

        save_audio_message_info_to_db(
            filename=filename, 
            user_id=message.from_user.id, 
            message_id=message.message_id, 
            emotion=emotion, 
            mfccs=mfccs, 
            chroma=chroma, 
            mel=mel, 
            contrast=contrast, 
            tonnetz=tonnetz
        )

        os.remove(filename)

    except Exception as e:
        print(f"Ошибка при обработке голосового сообщения: {e}")
        bot.reply_to(message, '''
Упс... Кажется, что-то пошло не так. 🙈 
Возможно, роботы захватили мир, или просто у меня случился небольшой сбой. 😅
Попробуй ещё раз позже!
''') 



# Обработчик текстовых сообщений
@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    try:
        text = message.text

        processed_text = preprocess_text(text)
        emotion = predict_emotion_from_text(processed_text)

        emoji = get_emoji_for_emotion(emotion)
        
        bot.reply_to(message, f"{emoji}")  
        print("Ответ пользователю отправлен.")

        save_text_message_info_to_db(message.from_user.id, message.message_id, text, processed_text, emotion)

    except Exception as e:
        print(f"Ошибка при обработке текстового сообщения: {e}")
        bot.reply_to(message, '''
Упс... Кажется, что-то пошло не так. 🙈 
Возможно, роботы захватили мир, или просто у меня случился небольшой сбой. 😅
Попробуй ещё раз позже!
''') 
        

# Запуск бота
if __name__ == '__main__':
    print("Бот запущен и ожидает сообщений...")
    bot.polling(none_stop=True)