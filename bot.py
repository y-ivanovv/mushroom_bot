import telebot
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
import logging

with open("token.txt", "r") as f:
    API_TOKEN = f.read().strip()

MODEL_PATH = 'mushroom_model'

bot = telebot.TeleBot(API_TOKEN)
logging.basicConfig(level=logging.INFO)

processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

def classify_mushroom(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    return label

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "👋 Привет! Отправь мне фото гриба, и я постараюсь распознать его.")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    temp_file_path = f"temp_{message.chat.id}.jpg"
    with open(temp_file_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    try:
        label = classify_mushroom(temp_file_path)
        bot.send_message(message.chat.id, f"🍄 Я думаю, это: *{label}*", parse_mode="Markdown")
    except Exception as e:
        logging.exception("Ошибка при распознавании:")
        bot.send_message(message.chat.id, "❌ Не удалось распознать изображение.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == '__main__':
    bot.infinity_polling()
