# import telebot
import sys
import os
import telebot

sys.path.append(os.path.abspath(os.path.join('Mask_RCNN')))
os.chdir('Mask_RCNN')

API_KEY = '2038935012:AAEUSpqHqqsg3JdRTBhhQR98xwMlph3JS7E'
bot = telebot.TeleBot(API_KEY)

info_message = 'Belum dimulai'
@bot.message_handler(commands=['info'])
def info(message):
    file = open("space.txt", "r")
    emptyspace = file.read()
    info_message = 'Halo! Berikut informasi ketersediaan parkir di Universitas Indonesia :\n1. Gedung Dekanat FTUI\n\t\t- Tersedia : {}'.format(emptyspace)
    bot.send_message(message.chat.id, info_message)

bot.polling()
while True:
    pass



