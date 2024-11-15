from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import os
import torch
import re
import emoji
import pymorphy2
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')



try:
    with open('config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Ошибка: Файл config.json не найден.")
    exit(1)
except json.JSONDecodeError:
    print("Ошибка: Не удалось декодировать config.json. Проверьте синтаксис файла.")
    exit(1)

# Путь к модели из config.json
model_path = os.path.join(config['text_model_path'])


if torch.cuda.is_available():
    device = 0  
else:
    device = -1  


# Загрузка модели и токенизатора:
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Создание pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)  


id2label = {
    0: "joy", 
    1: "sadness", 
    2: "surprise",
    3: "fear", 
    4: "anger", 
    5: "neutral"
} 

# Заменяем ASCII-смайлики на текстовые описания
def replace_ascii_emoticons(text):
    text = emoji.demojize(text) 
    text = re.sub(r':\w+:', lambda m: f' {m.group(0)[1:-1].upper()} ', text)
    return text

morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text, morph=morph): 
    try:
        text = replace_ascii_emoticons(text) 
        text = text.lower()
        text = re.sub(r'http\S+', '', text) 
        text = re.sub(r'<.*?>', '', text) 
        text = re.sub(r'@\w+', '', text) 
        text = re.sub(r'#', '', text)
        text = ''.join([i for i in text if not i.isdigit()])  
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9 ]', '', text) 
        words = text.split()
        words = [word for word in words if word not in stopwords.words('russian')] 
        words = [morph.parse(word)[0].normal_form for word in words]  
        processed_text = ' '.join(words)
        print("Текст обработан.")
        return processed_text
    except Exception as e:
        print(f"Ошибка при обработке текста: {e}")
        return None  


def predict_emotion_from_text(processed_text):
    try:
        result = classifier(processed_text)[0]
        predicted_label_id = int(result['label'].split('_')[1])
        predicted_emotion = id2label.get(predicted_label_id, "unknown")
        print("Эмоция из текста предсказана.")
        return predicted_emotion
    except Exception as e:
        print(f"Ошибка при распознавании эмоции из текста: {e}")
        return None