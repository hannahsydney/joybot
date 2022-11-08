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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Bidirectional
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_DETERMINISTIC_OPS']='1'
nltk.download("all")

class DepressionModelTraining:
    def __init__(self):
        self.df=pd.read_csv("data/depression_dataset_reddit_cleaned.csv")
        self.df.head()
        self.df["is_depression"].value_counts(normalize=True).plot(kind="bar")
        self.df.info()

    def preprocessing(self):
        # remove stop words
        w=WordNetLemmatizer()
        for i in range(len(self.df)):
            review=re.sub('[^a-zA-Z]', ' ', self.df["clean_text"][i])
            review=review.lower()
            review=review.split()
            review=[w.lemmatize(word) for word in review if not word in set(stopwords.words("english"))]
            review=" ".join(review)
            self.df["clean_text"][i]=review
        self.df.head()
        
        # get max sentence length
        s=set()
        for i in range(len(self.df)):
            k=self.df["clean_text"][i].split()
            for j in range(len(k)):
                s.add(k[j])
        voc_size=len(s)
        
        # get one hot encoding
        onehot_repr=[tf.keras.preprocessing.text.one_hot(words,voc_size)for words in self.df["clean_text"]]

        # get unique words count
        sent_length=0
        for i in onehot_repr:
            if len(i)>sent_length:
                sent_length=len(i)

        # word embeddings
        sent_length=self.calcMaxSentLength()
        embedded_docs=tf.keras.preprocessing.sequence.pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
        return embedded_docs
        
    def train(self):
        embedding_vector_features=self.calcMaxSentLength()*2
        model = model=tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(voc_size,embedding_vector_features,input_length=sent_length))
        model.add((tf.keras.layers.LSTM(100)))
        model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
        # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['sparse_categorical_accuracy'])
        print(model.summary())

        Y=self.df["is_depression"]
        
        embedded_docs = self.preprocessing()
        X_train,X_test,Y_train,Y_test=train_test_split(embedded_docs,Y,test_size=0.15,random_state=42,stratify=Y)
        model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=3,batch_size=16)

        Y_pred=model.predict(X_test)
        Y_pred=(Y_pred>=0.5).astype("int")

        print(classification_report(Y_test,Y_pred))
        print(confusion_matrix(Y_test,Y_pred))

        model.save_weights("model/model.h5")
    

def main():
    DepressionModelTraining()

if __name__ == '__main__':
    main()
