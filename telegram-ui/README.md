# JoyBot Telegram UI

This bot makes use of the [Telegram Bot API](https://github.com/python-telegram-bot/python-telegram-bot "Python-Telegram-Bot"). 


## Setup
Create a `.env` file under the `/src` directory. This file defines all the environment variables necessary for the app to run.

Necessary environment variables include:
- `bot_token`: The Telegram bot API token

## Run
To run the application locally:
- Create a virtual environment with `python -m venv venv`
- Activate the venv using `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac)
- Run `python src/joybot.py`