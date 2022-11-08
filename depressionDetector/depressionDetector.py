import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
nltk.download("all")


class Detector:
  def buildModel(self, voc_size, sent_length):
      embedding_vector_features=sent_length*2
      model = model=tf.keras.models.Sequential()
      model.add(tf.keras.layers.Embedding(voc_size,embedding_vector_features,input_length=sent_length))
      model.add((tf.keras.layers.LSTM(100)))
      model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
      model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
      model.load_weights("model/model.h5")
      return model

  def __init__(self):
    self.userInput = dict()
    self.userInputDf = pd.DataFrame(columns=['input', 'is_depressed'])
    self.score = 0
    # total unique voc size used to training model
    self.voc_size = 18611
    # longest sentence length used to train model
    self.sent_length = 1844
    self.model = self.buildModel(self.voc_size, self.sent_length)

  def updateScore(self):
    # score = # depressed input / # total input
    depressedCount = self.userInputDf[self.userInputDf['is_depressed'] == 1].shape[0]
    self.score = round(depressedCount/self.userInputDf.shape[0], 2)

  def getScore(self):
    return self.score

  def preprocessInput(self, input):
    # remove stop words
    w = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z]', ' ', input)
    review = review.lower()
    review = review.split()
    review = [w.lemmatize(word) for word in review if not word in set(
        stopwords.words("english"))]
    review = " ".join(review)
    inputWoStopWord = review

    # one hot encoding and word embedding
    input_onehot = tf.keras.preprocessing.text.one_hot(inputWoStopWord, self.voc_size)
    return tf.keras.preprocessing.sequence.pad_sequences([input_onehot], padding='pre', maxlen=1844)

  def depressionDetection(self, input):
    processedInput = self.preprocessInput(input)
    pred = self.model.predict(processedInput)
    pred = (pred >= 0.5).astype("int")
    self.userInput.update({input: pred[0][0]})
    self.userInputDf = self.userInputDf.append(
        {'input': input, 'is_depressed': pred[0][0]}, ignore_index=True)
    self.updateScore()

def init():
    return Detector()
