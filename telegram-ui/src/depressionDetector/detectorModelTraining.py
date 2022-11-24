import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
nltk.download("all")

load_dotenv()
DEPRESSION_MODEL_PATH = os.environ['depression_model_path']
TRAINING_DATA_PATH = os.environ['training_data_path']


class DepressionModelTraining:
    def __init__(self):
        self.df = pd.read_csv(TRAINING_DATA_PATH)
        self.df.head()
        self.df["is_depression"].value_counts(normalize=True).plot(kind="bar")
        self.df.info()
        self.voc_size = 0
        self.sent_length = 0

    def preprocessing(self):
        # remove stop words
        w = WordNetLemmatizer()
        for i in range(len(self.df)):
            review = re.sub('[^a-zA-Z]', ' ', self.df["clean_text"][i])
            review = review.lower()
            review = review.split()
            review = [w.lemmatize(word) for word in review if not word in set(
                stopwords.words("english"))]
            review = " ".join(review)
            self.df["clean_text"][i] = review

        # get max sentence length
        s = set()
        for i in range(len(self.df)):
            k = self.df["clean_text"][i].split()
            for j in range(len(k)):
                s.add(k[j])
        voc_size = len(s)
        self.voc_size = voc_size

        # get one hot encoding
        onehot_repr = [tf.keras.preprocessing.text.one_hot(
            words, voc_size)for words in self.df["clean_text"]]

        # get unique words count
        sent_length = 0
        for i in onehot_repr:
            if len(i) > sent_length:
                sent_length = len(i)
        self.sent_length = sent_length

        # word embeddings
        embedded_docs = tf.keras.preprocessing.sequence.pad_sequences(
            onehot_repr, padding='pre', maxlen=sent_length)
        return embedded_docs

    def train(self):
        print("Preprocessing Data....")
        embedded_docs = self.preprocessing()

        print("Training model....")
        embedding_vector_features = self.sent_length*2
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(self.voc_size,
                  embedding_vector_features, input_length=self.sent_length))
        model.add((tf.keras.layers.LSTM(100)))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['binary_accuracy'])
        # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['sparse_categorical_accuracy'])
        # print(model.summary())

        Y = self.df["is_depression"]

        X_train, X_test, Y_train, Y_test = train_test_split(
            embedded_docs, Y, test_size=0.15, random_state=42, stratify=Y)
        model.fit(X_train, Y_train, validation_data=(
            X_test, Y_test), epochs=10, batch_size=16)

        Y_pred = model.predict(X_test)
        Y_pred = (Y_pred >= 0.5).astype("int")

        # print(classification_report(Y_test, Y_pred))
        # print(confusion_matrix(Y_test, Y_pred))

        model.save_weights(DEPRESSION_MODEL_PATH)
        print("Training completed and model saved!")


def init():
    return DepressionModelTraining()


# if __name__ == '__main__':
#     main()
