import os
import cv2
import requests
import numpy as np
from keras import models
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from urllib.request import urlopen
from token_bot import token

# URLs
url = "https://api.telegram.org/bot/".replace("bot", "bot"+token)
url_file = "https://api.telegram.org/file/bot/".replace("bot", "bot"+token)

# Path
path_model = 'model/'

# Load Model
json_file = open(path_model+'model.json', 'r')
model_json = json_file.read()
json_file.close()
model = models.model_from_json(model_json)
model.load_weights(path_model+'model.h5')

# Compile Model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Functions
def start(update, context):
    """ Welcome info for users """

    # User Informations
    user_info = update.message.from_user
    username = user_info['username']

    # Create Message
    message_text = 'Welcome!'
    message_text += '\n'
    message_text += 'This Bot classifies facial emotions based on images.'
    message_text += '\n'
    message_text += 'Say "hello" \N{winking face}'
    
    # Send Message
    update.message.reply_text(message_text)


def help(update, context):
    """Standart helper funtion to give the user some advices."""

    # User Informations
    user_info = update.message.from_user
    username = user_info['username']

    # Create Message
    message_text = 'Informations:'
    message_text += '\n'
    message_text += '1) Only one face per photo.'
    message_text += '\n'
    message_text += '2) The face should be in the center of the photo.'

    # Send Message
    update.message.reply_text(message_text)


def repeater(update, context):
    """ Repeater for text messages """

    # User Informations
    user_info = update.message.from_user
    language_code = user_info['language_code']
    username = user_info['username']

    # Define Messages
    message_text = 'Hello ' + username
    message_text += '\n'
    message_text += 'Write/Click /help for more informations or load up a photo.'
    
    # Send Message
    update.message.reply_text(message_text)

def image_preprocessing(image, image_size):
    """ Image Preprocessing """

    # Load Image
    readFlag=cv2.COLOR_BGR2GRAY
    image_gray = cv2.cvtColor(image, readFlag)
    
    # Crop Image
    mid_row = int(image_gray.shape[0]/2)
    mid_col = int(image_gray.shape[1]/2)
    if image_gray.shape[0]>image_gray.shape[1]:
        image_cropped = image_gray[mid_row-mid_col:mid_row+mid_col,
                                   0:image_gray.shape[1]]
    else:
        image_cropped = image_gray[0:image_gray.shape[0],
                                   mid_col-mid_row:mid_col+mid_row]
    
    # Rescale Image
    image_rescale = cv2.resize(image_cropped,
                               dsize=(image_size, image_size),
                               interpolation=cv2.INTER_AREA)
    return image_rescale 


def predict_facial_emotions(update, context):
    """ Function to load, prepare the image and predict the facial to emotion."""

    # User Informations
    user_info = update.message.from_user
    username = user_info['username']

    # Image Size
    image_size = 48

    emotions = {0: 'Angry',
                1: 'Disgust',
                2: 'Fear',
                3: 'Happy',
                4: 'Sad',
                5: 'Surprise',
                6: 'Neutral'}
    
    # Use the image with highest resolution
    image_file = update.message.photo[-1]
    
    # Extract File Path
    getFile = requests.get(url+"getFile"+"?file_id="+image_file.file_id)
    file_path = getFile.json()['result']['file_path']

    # Extract Message Attachment
    readFlag=cv2.COLOR_BGR2GRAY
    resp = urlopen(url_file+file_path)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    
    image_rescale = image_preprocessing(image, image_size)

    # Define Model Input
    image_array = np.zeros(shape=(1, image_size, image_size))
    image_array[0] = image_rescale
    images = image_array.reshape((image_array.shape[0], image_size, image_size, 1))
    images = images.astype('float32')/255

    # Predict Image
    labels = model.predict(images)
    labels = labels.reshape(len(emotions))

    # Define Message
    message_text = 'Results:\n'
    for i in range(len(emotions)):
        message_text += str(int(100*labels[i])).zfill(2)+'% '+emotions[i]+'\n'

    # Send Message
    update.message.reply_text(message_text)


def main():

    """ TEST BOT CONNECTION"""
    resp = requests.get(url+'getMe')
    print("staus code:", resp.status_code)

    # Define Updater
    updater = Updater(token, use_context=True)

    # Define Dispatcher
    dispatcher = updater.dispatcher

    # Define Handler
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('help', help))
    dispatcher.add_handler(MessageHandler(Filters.text, repeater))
    dispatcher.add_handler(MessageHandler(Filters.photo, predict_facial_emotions))
    
    # start_polling actually starts the bot
    updater.start_polling()

    # idle block the script until the user sends a command
    updater.idle()



if __name__=='__main__':
    main()

