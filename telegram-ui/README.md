# JoyBot Telegram UI

This bot makes use of the [Telegram Bot API](https://github.com/python-telegram-bot/python-telegram-bot "Python-Telegram-Bot"). 


## Setup
Modifiy the `.env` file under the `/src` directory. This file defines all the environment variables necessary for the app to run.

Necessary environment variables include:
- `bot_token`: The Telegram bot API token

## Run JoyBot
To run the application locally:
- Create a virtual environment with `python -m venv venv`
- Activate the venv using `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac)
- Run `python src/joybot.py`

## Re-train Depression Detection Model
The depression detection model is already trained. To re-train the depression detection model:
- Run `python src/trainModel.py`
