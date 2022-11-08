import os
import logging
from time import sleep
from typing import Dict

from dotenv import load_dotenv
from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )

from telegram import ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from util.errorhandler import error_handler
from communicator.communicator import Communicator
from queue import Queue


# *=================================== SETUP ===================================*
load_dotenv()

TOKEN = os.environ['bot_token']
PORT = int(os.environ.get('PORT', '8443'))

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

communicator = Communicator()

CHAT, NAME, CONFIRMED_NAME, GREETED, TESTLOOP = range(5)


# *=================================== CHAT FUNCTIONS ===================================*

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['communicator'] = Communicator(max_eval_inputs=10)
    # communicator.__init__()
    """Start the conversation"""
    await update.message.reply_text('Hi, nice to meet you!')

    # reply with chat action "Typing..."
    await fake_typing(update)
    # await update.message.reply_chat_action(ChatAction.TYPING)
    # sleep(0.5)
    await update.message.reply_text('May I know your name?')

    return NAME


async def confirm_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Ask for the user's name and confirm"""
    name = update.message.text
    context.user_data['name'] = name

    options = [['Yes', 'No']]

    await fake_typing(update)
    await update.message.reply_text(
        'Can I call you ' + name + '?',
        reply_markup=ReplyKeyboardMarkup(
            options, one_time_keyboard=True, input_field_placeholder='Is this your name?'
        ),
    )

    return CONFIRMED_NAME


async def wrong_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Wrong name handling"""
    await fake_typing(update)
    await update.message.reply_text("What's your name?")

    return NAME


async def greet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Greet user for the first time and invoke first communicator call"""
    name = context.user_data["name"]

    await fake_typing(update)
    await update.message.reply_text((
        f'That\'s a lovely name, {name}.'
    ))

    await fake_typing(update)
    # start the bot
    communicator = context.user_data['communicator']
    message = communicator.start()

    await update.message.reply_text(message)

    # context.user_data['first_prompt'] = True
    # initialise message history
    context.chat_data['message_history'] = Queue(maxsize=5)

    return GREETED


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recursive chat function until communicator ends the chat"""
    reply = update.message.text

    # update the chat history
    if context.chat_data['message_history'].full():
        context.chat_data['message_history'].get()
    context.chat_data['message_history'].put(reply)

    history = context.chat_data['message_history']
    
    communicator = context.user_data['communicator']
    bot_response = communicator.handle_input(reply, list(history.queue))

    # await update.message.reply_text(str(list(history.queue)))

    # probably won't be used
    if bot_response == "end":
        await test_loop(update, context)
        return ConversationHandler.END

    await update.message.reply_text(bot_response)

    return CHAT


async def bye(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await fake_typing(update)
    await update.message.reply_text('Well then, it was nice talking to you.')

    await fake_typing(update)
    await update.message.reply_text('See you again soon!')

    return ConversationHandler.END


async def test_loop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        'You have reached the test loop.\n'
        'The bot will now stop. '
        'Restart it again with /start.'
    )

    return ConversationHandler.END


# *=================================== MISCELLANEOUS ===================================*

async def fake_typing(update: Update):
    await update.message.reply_chat_action(ChatAction.TYPING)
    sleep(1)
    return


def facts_to_str(user_data: Dict[str, str]) -> str:
    # """Helper function for formatting the gathered user info."""
    facts = [f'{key} - {value}' for key, value in user_data.items()]
    return "\n".join(facts).join(['\n', '\n'])


async def show_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # """Display the gathered info."""
    await update.message.reply_text(
        f"This is what you already told me: {facts_to_str(context.user_data)}"
    )


# *=================================== MAIN ===================================*

def main() -> None:
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, confirm_name)],
            CONFIRMED_NAME: [
                MessageHandler(filters.Regex('^Yes$'), greet),
                MessageHandler(filters.Regex('^No$'), wrong_name),
            ],
            GREETED: [MessageHandler(filters.TEXT & ~filters.COMMAND, chat)],
            CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, chat)],
            TESTLOOP: [MessageHandler(filters.TEXT & ~filters.COMMAND, test_loop)],
        },
        fallbacks=[
            CommandHandler('bye', bye),
            CommandHandler('start', start),
        ],
        name="joybot_convo",
    )

    application.add_handler(conv_handler)

    show_data_handler = CommandHandler('show_data', show_data)
    application.add_handler(show_data_handler)
    application.add_error_handler(error_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
