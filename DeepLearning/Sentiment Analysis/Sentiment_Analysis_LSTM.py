# -*- coding: utf-8 -*-
"""

@author: Sreenivas.J
"""
#https://www.kaggle.com/c/word2vec-nlp-tutorial
#Bag of Words Meets Bags of Popcorn

import os
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup #Need to install this explicitly
from nltk import word_tokenize          
from nltk.corpus import stopwords
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

data_dir = 'D:/Data Science/Data/sentiment-analysis'
glove_file = 'D:/Data Science/Data//glove.6B/glove.6B.50d.txt'

word_embed_size = 50 #It should be equal to Glove dimension
batch_size = 32
epochs = 2
seq_maxlen = 80

def loadGloveWordEmbeddings(glove_file):
    embedding_vectors = {}
    f = open(glove_file,encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embedding_vectors[word] = value
    f.close()
    return embedding_vectors

def cleanReview(review):        
        #1.Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #print(review_text)
        #2.Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #print(review_text)
        #3.Convert words to lower case
        review_text = review_text.lower()
        #print(review_text)
        #4.remove stop words
        review_words = word_tokenize(review_text)
        words = [w for w in review_words if not w in stopwords.words("english")]
        #print(words)
        return ' '.join(words)
    
def buildVocabulary(reviews):
    tokenizer = Tokenizer(lower=False, split=' ')
    tokenizer.fit_on_texts(reviews)
    return tokenizer

def getSequences(reviews, tokenizer, seq_maxlen):
    reviews_seq = tokenizer.texts_to_sequences(reviews)
    return np.array(pad_sequences(reviews_seq, maxlen=seq_maxlen))

def getEmbeddingWeightMatrix(embedding_vectors, word2idx):    
    embedding_matrix = np.zeros((len(word2idx)+1, word_embed_size))
    for word, i in word2idx.items():
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

imdb_train = pd.read_csv(os.path.join(data_dir,"labeledTrainData.tsv"), header=0, 
                    delimiter="\t", quoting=3)
imdb_train.shape
imdb_train.info()
imdb_train.loc[0:4,'review']

#preprocess text
review_train_clean = imdb_train['review'][0:4].map(cleanReview)
print(review_train_clean[0])


#build vocabulary over all reviews
tokenizer = buildVocabulary(review_train_clean)
vocab_size = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(vocab_size)

X_train = getSequences(review_train_clean, tokenizer, seq_maxlen)
y_train = np_utils.to_categorical(imdb_train['sentiment'][0:4])

#load pre-trained word embeddings
embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))
#get embedding layer weight matrix
embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer.word_index)
print(embedding_weight_matrix.shape)

#build model        
input = Input(shape=(X_train.shape[1],))

EmbedReview = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=seq_maxlen, weights=[embedding_weight_matrix], 
                   trainable = False) (input)
inner = Bidirectional(LSTM(100))(EmbedReview)
inner = Dropout(0.3)(inner)
inner = Dense(50, activation='relu')(inner)
output = Dense(2, activation='softmax')(inner)

model = Model(inputs = input, outputs = output)
print(model.summary())
model.compile(Adam(lr=0.01), loss = 'binary_crossentropy', metrics=['accuracy'])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train, verbose=1, epochs=epochs, batch_size=batch_size, 
                    callbacks=[save_weights], validation_split=0.1)
os.getcwd()
imdb_test = pd.read_csv(os.path.join(data_dir,"testData.tsv"), header=0, 
                    delimiter="\t", quoting=3)
imdb_test.shape
imdb_train.shape
imdb_test.info()
#imdb_test.loc[0:4,'review']

#preprocess text
review_test_clean = imdb_test['review'].map(cleanReview)
#review_test_clean = imdb_test['review'][0:4].map(cleanReview)
print(len(review_test_clean))

X_test = getSequences(review_test_clean, tokenizer, seq_maxlen)
imdb_test['sentiment'] = model.predict(X_test).argmax(axis=-1)
imdb_test.to_csv(os.path.join(data_dir,'submission.csv'), columns=['id','sentiment'], index=False)
