import telebot
from telebot import types
from time import sleep
import logging
from pymongo import MongoClient

from network import Texliven
from config import prod_telegram_token, telegram_token, db_adress

mongo = MongoClient(db_adress)
db = mongo.texliven

DEFAULT = "DEFAULT"
RIGHT_ANSWER = "RIGHT_ANSWER"


logger = logging.getLogger("TexLiven")
hdlr = logging.FileHandler('log/TexLiven.log', encoding = "UTF-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

message_log = "{chat_id} >>> {question} >>>>>>> {answer}"

TexLivenFront = telebot.TeleBot(prod_telegram_token, threaded=False)
TexLivenBack = telebot.TeleBot(telegram_token)

TexLivenMind = Texliven()
print("Ready? Go!")
logger.info("Bot is started")

offset = 0
offset_back = 0
stop_all = False

question_answer = {}
state = {}

@TexLivenFront.message_handler(commands=['help', 'start'])
def help_start(message):
    TexLivenFront.send_message(message.chat.id, """(Системное сообщение) Просто
            пиши мне""")

@TexLivenFront.message_handler(commands=["default", "help_dataset"])
def help_teatch(message):
    user = db.user_state.find_one({"chat-id": message.chat.id})
    if message.text == "/default":
        if user is not None:
            user["state"] = "default"
            db.user_state.save(user)
        else:
            user = {"chat-id": message.chat.id, "state": "default"}
            db.user_state.insert_one(user)
    else:
        if user is not None:
            user["state"] = "dataset"
            db.user_state.save(user)
        else:
            user = {"chat-id": message.chat.id, "state": "dataset"}
            db.user_state.insert_one(user)
    TexLivenFront.send_message(message.chat.id, "Готово")


@TexLivenFront.message_handler(content_types=["text"])
def answer(message):
    user_st = db.user_state.find_one({"chat-id": message.chat.id})
    print(user_st)
    print(message.chat.id)
    if user_st is not None and user_st["state"] == "dataset":
        us_state = state.get(message.chat.id, DEFAULT)
        user = question_answer.get(message.chat.id, {"message_id": None,
                                                "question": None})

        if us_state == RIGHT_ANSWER:
            messag = db.message.find_one({"message_id": "{}/{}".format(
                                                            message.chat.id,
                                                            user["message_id"])})
            if messag is not None:
                messag["right-answer"] = message.text
                db.message.save(messag)
            else:
                db.message.insert_one({"question": user["question"],
                                        "answer": user["answer"],
                                        "message_id": "{}/{}".format(
                                                                message.chat.id,
                                                                user["message_id"]),
                                        "right-answer": message.text})
            try:
                with open("log/dia2.log", "a") as fp:
                    fp.write("{0} > {1} >> {2}\n".format(user["message_id"], user["question"], message.text))
            except Exception:
                pass
            TexLivenFront.send_message(message.chat.id, "Спасибо за помощь :)\nПродолжаем")
            state[message.chat.id] = DEFAULT

        else:
            quest = str(message.text)
            try:
                messages = TexLivenMind.generate_answer(quest, t=1)
            except Exception:
                return None
            if messages == "":
                messages = " "
            logger.info(message_log.format(chat_id=message.chat.id,
                                            question=message.text,
                                            answer=messages))

            question_answer[message.chat.id] = {"question": quest,
                                                "answer": messages,
                                                "message_id": message.message_id}
            db.message.insert_one({"question": quest,
                                    "answer": messages,
                                    "message_id": "{}/{}".format(
                                                            message.chat.id,
                                                            message.message_id)})

            keyboard = types.InlineKeyboardMarkup()
            like = types.InlineKeyboardButton(text=u'\U0001F44D',
                                                        callback_data="like")
            dislike = types.InlineKeyboardButton(text=u'\U0001F44E',
                                                         callback_data="dislike")
            keyboard.add(like)
            keyboard.add(dislike)


            TexLivenFront.send_message(message.chat.id, messages,
                                        reply_markup=keyboard)
    else:
        quest = str(message.text)
        messages = TexLivenMind.generate_answer(quest, t=1)
        if messages == "":
            messages = " "
        logger.info(message_log.format(chat_id=message.chat.id,
                                        question=message.text,
                                        answer=messages))


        TexLivenFront.send_message(message.chat.id, messages)

@TexLivenFront.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    user = question_answer.get(call.message.chat.id, {"message_id": None,
                                                        "question": None,
                                                        "answer": None})

    if call.data in ["like", "dislike"]:
        message = db.message.find_one({"message_id": "{}/{}".format(
                                                        call.message.chat.id,
                                                        user["message_id"])})
        if message is not None:
            message["assessment"] = call.data
            db.message.save(message)
        else:
            db.message.insert_one({"question": user["question"],
                                    "answer": user["answer"],
                                    "message_id": "{}/{}".format(
                                                            call.message.chat.id,
                                                            user["message_id"]),
                                    "assessment": call.data})
        with open("log/dia1.log", "a") as fp:
            fp.write("{} > {} >>> {} >> {}\n".format(user["message_id"],
                                                user["question"],
                                                user["answer"],
                                                call.data))

        TexLivenFront.edit_message_text(chat_id=call.message.chat.id,
                                        message_id=call.message.message_id,
                                        text=call.message.text)

        if call.data == "dislike":
            keyboard = types.InlineKeyboardMarkup()
            yes = types.InlineKeyboardButton(text="Да", callback_data="yes")
            no = types.InlineKeyboardButton(text="Нет", callback_data="no")
            keyboard.add(yes)
            keyboard.add(no)
            TexLivenFront.send_message(call.message.chat.id,
                    "Вы бы хотели исправить мой ответ?",
                    reply_markup=keyboard)

    else:
        if call.data == "yes":
            state[call.message.chat.id] = RIGHT_ANSWER
            TexLivenFront.edit_message_text(chat_id=call.message.chat.id,
                                            message_id=call.message.message_id,
                                            text="Введите правильный ответ")
        else:
            TexLivenFront.edit_message_text(chat_id=call.message.chat.id,
                    message_id=call.message.message_id, text="Жаль:(\nПродолжаем")

TexLivenFront.polling(none_stop=True)

#    updates_back = TexLivenBack.get_updates(offset=offset_back, timeout=1)
#    if len(updates_back) != 0:
#        for update in updates_back:
#            if update.message.chat.username == "ADmitri" and update.message.text == "/stop":
#                TexLivenBack.send_message(update.message.chat.id, "Server is stopped")
#                offset_back = update.update_id + 1
#                TexLivenBack.get_updates(offset=offset_back, timeout=1)
#                logger.info("Server is stopped")
#                exit()
