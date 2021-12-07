# README
Welcome to the facial expression telegram bot app. Based on a facial emotion [dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) we developed a telegram bot to classify the facial emotion of an image.

The bot is called FacialExpressioBot.

For details we recommend this kaggle [notebook](https://www.kaggle.com/drcapa/facial-expression-telegram-bot/).

Please notice: To run the app it is necessary to create the file token_pot.py with your telegram bot token **XXX** like this:
```python
token = "XXX"
```

The app file app.py expects that file to use

```python
from token_bot import token
```

This file is not included in the repository. How to create a telegram bot token is also described in the recommended notebook.