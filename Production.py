from network import Texliven
import telebot
import config

texliven = Texliven()
bot = telebot.TeleBot(config.production_telegram_token)

@bot.message_handler(content_types=["text"])
def ansewer_for_message(message): # Название функции не играет никакой роли, в принципе
    bot.send_message(message.chat.id, texliven.generate_answer(message.text, t=3))
    
if __name__ == '__main__':
    bot.polling(none_stop=True)

