import os
import re

import numpy as np
import keras

from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import preprocess_input, decode_predictions

from aiogram import Bot, Dispatcher, executor, types
import aiohttp
import logging


TOKEN = os.getenv('TELEGRAM_API_TOKEN_WHATABOT')

model = keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    _user_name = message.from_user.first_name
    _text = "Хэлло, бой! Твоё имя - %s!\nА меня ты можешь называть: What_is_this_bot?!" %_user_name
    await message.reply(_text)

@dp.message_handler(content_types=['text'])
async def handle_docs_photo(message):
    _user_name = message.from_user.first_name
    _text = "Смотри, %s! Ты отправляешь фотографию, а я говорю тебе: 'what is this' and % of prediction score" %_user_name

    await bot.send_message(message.chat.id, _text)


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
    _user_name = message.from_user.first_name
    _user_id = message.from_user.id
    _text = "Спасибо, %s! Подожди секунду..." %_user_name
    _photo_name = './photos/carphoto_%s.jpg' %_user_id
    await message.photo[-1].download(_photo_name),
    await bot.send_message(message.chat.id, _text)
    _img_path = _photo_name
    _img = load_img(_img_path, target_size=(299, 299))  # this is a PIL image
    _img_arr = img_to_array(_img)  # Numpy array with shape (299, 299, 3)
    _img_arr = _img_arr.reshape((1,) + x.shape)  # Numpy array with shape (1, 299, 299, 3)
    # Rescale by 1/255
    _img_arr /= 255

    _preds = model.predict(_img_arr)
    _pred_decoded = decode_predictions(preds)[0][0:3]
    _pred_decoded = [el[1:] for el in _pred_decoded]
    _test_text_of_prediction = ' '.join([str(elem) for elem in _pred_decoded])
    _test_text_of_prediction = re.sub("[,']", "", _test_text_of_prediction)
    _test_text_of_prediction = re.sub("[()]", "\n", _test_text_of_prediction)

    await bot.send_message(message.chat.id, _test_text_of_prediction)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
